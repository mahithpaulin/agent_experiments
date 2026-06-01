import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineActivation(nn.Module):
    def __init__(self, degree=3, n_knots=10, knot_range=(-2.0, 2.0)):
        super().__init__()
        self.degree = degree
        self.n_knots = n_knots
        self.knot_range = knot_range
        # Initialize knots uniformly in range
        knots = torch.linspace(knot_range[0], knot_range[1], n_knots)
        # Make knots learnable parameters
        self.knots = nn.Parameter(knots)
        # Initialize coefficients for B-spline basis functions, one per knot
        # We will learn coefficients per edge in KANLayer, so here just placeholder
        # This module will be used per edge with separate coeffs
        # So here we only provide basis evaluation method

    def forward(self, x, coeffs):
        # x: (...), coeffs: (n_knots,)
        # Evaluate B-spline basis functions at x for given knots and degree
        # Then output = sum(coeffs_i * basis_i(x))
        basis = self.bspline_basis(x)
        # basis shape (..., n_knots)
        # coeffs shape (n_knots,)
        # output shape (...)
        out = (basis * coeffs).sum(dim=-1)
        return out

    def bspline_basis(self, x):
        # Cox-de Boor recursion formula for B-spline basis
        # x: (...), knots: (n_knots,)
        # Output: (..., n_knots)
        device = x.device
        knots = self.knots
        degree = self.degree
        n_knots = self.n_knots

        # Pad knots for degree
        # For clamped B-spline, pad start and end knots with multiplicity degree+1
        t = knots
        t = torch.cat([
            t[0].repeat(degree + 1),
            t,
            t[-1].repeat(degree + 1)
        ])
        # Number of basis functions = n_knots + degree
        n_basis = n_knots + degree

        # Initialize zero degree basis functions N_i,0(x)
        # N_i,0(x) = 1 if t_i <= x < t_{i+1}, else 0
        # For last knot interval include right endpoint
        x_exp = x.unsqueeze(-1)  # (...,1)
        N = ((x_exp >= t[:-1]) & (x_exp < t[1:])).float()
        # Fix last knot interval to include right endpoint
        last_interval = (x_exp == t[-1]).float()
        N[..., -1] += last_interval.squeeze(-1)

        # Recursion for degree d=1..degree
        for d in range(1, degree + 1):
            left_num = x_exp - t[:-d-1]
            left_den = t[d:-1] - t[:-d-1]
            left = torch.zeros_like(N)
            mask = left_den > 1e-8
            left[..., :-1] = torch.where(mask, (left_num / left_den) * N[..., :-1], torch.zeros_like(left_num))

            right_num = t[d+1:] - x_exp
            right_den = t[d+1:] - t[1:-d]
            right = torch.zeros_like(N)
            mask = right_den > 1e-8
            right[..., 1:] = torch.where(mask, (right_num / right_den) * N[..., 1:], torch.zeros_like(right_num))

            N = left + right

        # N shape (..., n_basis), but we want exactly n_knots basis functions
        # We return first n_knots basis functions (clamped)
        return N[..., :n_knots]


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_knots=10, degree=3, knot_range=(-2, 2)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_knots = n_knots
        self.degree = degree
        self.knot_range = knot_range

        # For each output dimension, we have input_dim inner functions phi_i_j
        # Each phi_i_j is a B-spline activation with learnable coeffs per edge
        # So coeffs shape: (output_dim, input_dim, n_knots)
        self.coeffs = nn.Parameter(torch.randn(output_dim, input_dim, n_knots) * 0.1)

        # Shared B-spline activation module (knots shared)
        self.bspline = BSplineActivation(degree=degree, n_knots=n_knots, knot_range=knot_range)

        # Outer linear psi: maps input_dim to output_dim
        self.psi = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        # x: (batch, input_dim)
        batch = x.shape[0]
        # Compute phi_i_j(x_j) for each output_dim i and input_dim j
        # We vectorize over batch and output_dim
        # x_j shape (batch,) for each j
        # coeffs shape (output_dim, input_dim, n_knots)
        # We want output shape (batch, output_dim, input_dim)
        phi = []
        for i in range(self.output_dim):
            # For output dim i, get coeffs (input_dim, n_knots)
            coeffs_i = self.coeffs[i]  # (input_dim, n_knots)
            # For each input dim j, apply bspline to x[:,j] with coeffs_i[j]
            phi_i = []
            for j in range(self.input_dim):
                x_j = x[:, j]  # (batch,)
                coeffs_ij = coeffs_i[j]  # (n_knots,)
                val = self.bspline(x_j, coeffs_ij)  # (batch,)
                phi_i.append(val.unsqueeze(-1))
            # Concatenate over input_dim
            phi_i = torch.cat(phi_i, dim=-1)  # (batch, input_dim)
            phi.append(phi_i.unsqueeze(1))  # (batch,1,input_dim)
        # Concatenate over output_dim
        phi = torch.cat(phi, dim=1)  # (batch, output_dim, input_dim)

        # Sum over input_dim for each output_dim: sum_j phi_i_j(x_j)
        sum_phi = phi.sum(dim=2)  # (batch, output_dim)

        # Apply outer linear psi to x: (batch, input_dim) -> (batch, output_dim)
        psi_out = self.psi(x)  # (batch, output_dim)

        # Final output: sum_phi + psi_out
        out = sum_phi + psi_out
        return out


class KAN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1,
                 n_knots=10, degree=3, knot_range=(-2, 2)):
        super().__init__()
        # One hidden KANLayer + final linear
        self.kan1 = KANLayer(input_dim, hidden_dim, n_knots, degree, knot_range)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.kan1(x)
        h = torch.tanh(h)
        out = self.out(h)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
