"""Explicit, method-independent measurement-cost model.

Unit: one distinct circuit (each run with the same shot budget S). Per quantity:
  energy <H>             : g(H) circuits             (qubit-wise-commuting groups of H)
  <H^2>                  : g(H^2) circuits
  V_l  (or dE/dtheta_l)  : |G_l| * g(H) Hadamard-test circuits  (one per Pauli string of G_l per group)
  M_ij (i <= j)          : |G_i| * |G_j| Hadamard-test circuits
CNOT cost of a circuit = CNOTs of the ansatz prefix it contains (first-order product of string
exponentials, 2(w-1) CNOTs per weight-w string) + CNOTs of the controlled Paulis (weight each).
This is an upper-bound bookkeeping applied identically to every method; translation-symmetry
savings (a factor ~N for the M and V entries) are available to all methods and not applied.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .paulis import PauliSum, qwc_groups


@dataclass
class CostModel:
    gH: int
    gH2: int

    @classmethod
    def for_hamiltonian(cls, H: PauliSum) -> "CostModel":
        return cls(qwc_groups(H), qwc_groups(H @ H))


@dataclass
class Ledger:
    model: CostModel
    circuits: float = 0.0
    cnots: float = 0.0
    by_kind: dict = field(default_factory=dict)

    tag: str = None             # if set, overrides the kind in by_kind (e.g. "screen", "opt")

    def _add(self, kind, n_circ, cnot_each):
        self.circuits += n_circ
        self.cnots += n_circ * cnot_each
        kind = self.tag or kind
        self.by_kind[kind] = self.by_kind.get(kind, 0.0) + n_circ

    def tagged(self, tag):
        """Context manager: attribute circuits to `tag` (bookkeeping only; totals are unchanged)."""
        led = self

        class _T:
            def __enter__(self_):
                self_.old, led.tag = led.tag, tag

            def __exit__(self_, *a):
                led.tag = self_.old
        return _T()

    @staticmethod
    def _prefix_cnot(layers, upto=None):
        ls = layers if upto is None else layers[:upto]
        return sum(L.cnot for L in ls)

    def energy(self, layers):
        self._add("energy", self.model.gH, self._prefix_cnot(layers))

    def h2(self, layers):
        self._add("H2", self.model.gH2, self._prefix_cnot(layers))

    def V_entry(self, layers, l, gen_layer=None):
        G = gen_layer if gen_layer is not None else layers[l]
        w = np.mean(G.weights)
        self._add("V", G.n_strings * self.model.gH, self._prefix_cnot(layers) + w)

    def M_entry(self, layers, i, j, Gi=None, Gj=None):
        Gi = Gi if Gi is not None else layers[i]
        Gj = Gj if Gj is not None else layers[j]
        w = np.mean(Gi.weights) + np.mean(Gj.weights)
        self._add("M", Gi.n_strings * Gj.n_strings, self._prefix_cnot(layers) + w)

    def vite_step(self, layers, with_h2=True):
        """McLachlan system: E, all V_l, all M_ij (i<=j); <H^2> only if the residual is needed."""
        self.energy(layers)
        if with_h2:
            self.h2(layers)
        for l in range(len(layers)):
            self.V_entry(layers, l)
            for m in range(l, len(layers)):
                self.M_entry(layers, l, m)

    def growth_column(self, layers, cand):
        """Extra entries needed to evaluate the residual with `cand` appended: V_c, M_cj, M_cc."""
        self.V_entry(layers, None, gen_layer=cand)
        for l in range(len(layers)):
            self.M_entry(layers, l, None, Gj=cand)
        self.M_entry(layers, None, None, Gi=cand, Gj=cand)

    def gradient_entry(self, layers, cand):
        self.V_entry(layers, None, gen_layer=cand)

    def snapshot(self):
        return self.circuits, self.cnots
