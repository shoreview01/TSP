import numpy as np

NEG = -1e12  # numerical -inf


class TSPHypercubeBCJR:
    """
    PDF 원래 식 기반으로 다시 정리한 BCJR + SOVA 버전.

    - 단일 trellis ψ[t, mask, last]만 사용 (with-β)
    - α_t(a)는 항상 ψ_{t-1}에서 직접 계산:
        α_t(a) = max_{m_{t-1}, last ∈ m_{t-1}}
                    [ ψ_{t-1}(m_{t-1}, last) + s(last, a) ]
      (여기에는 β_t는 안 들어가고, 과거 시점 ≤ t-1의 β 정보만 포함)
    - trellis → X_{it} 메시지:
        ζ_{it}(a) = α_t(a) + β_t(a) - λ_{it}(a)
      (∑_{i'≠i} λ_{i't}(a) = β_t(a) - λ_{it}(a) 사용한 형태)
    - ψ는 with-β forward trellis:
        ψ_t(m_t, a_t) = max_{m_{t-1}, a_{t-1}}
            [ ψ_{t-1}(m_{t-1}, a_{t-1}) + s(a_{t-1}, a_t) + β_t(a_t) ]
      (코드에서는 t=1..T, β_t(a_t)를 beta[t-1, a_t]에 매핑)
    - BCJR backward bwd[t, mask, last]와 함께
      Γ_t(m_t,last) = ψ_t(m_t,last) + bwd_t(m_t,last)로 SOVA LLR 계산.
    """

    def __init__(self, D, start_city=None,
                 damping=0.3, iters=200, verbose=False,
                 tiny_tiebreak=False, seed=0,
                 patience_no_cost_change=10, cost_tol=1e-12,
                 kappa_bcjr=1.0,
                 damp_L=0.5, damp_beta=0.5, damp_zeta=0.5):

        D = np.array(D, dtype=float)
        assert D.shape[0] == D.shape[1], "D must be square"
        C = D.shape[0]

        if start_city is None:
            start_city = 0
        start_city = int(start_city)
        assert 0 <= start_city < C

        # permute: start -> last (internal depot)
        perm = np.arange(C)
        if start_city != C - 1:
            perm[start_city], perm[C - 1] = perm[C - 1], perm[start_city]
        inv_perm = np.empty(C, dtype=int)
        inv_perm[perm] = np.arange(C)

        self.orig_D = D
        self.D = D[perm][:, perm]
        self.perm = perm
        self.inv_perm = inv_perm

        self.C = C
        self.N = C - 1
        self.depot = C - 1

        self.verbose = verbose
        self.damp = float(damping)
        self.iters = int(iters)
        self.tiny_tiebreak = bool(tiny_tiebreak)
        self.rng = np.random.default_rng(seed)
        self.patience_no_cost_change = int(patience_no_cost_change)
        self.cost_tol = float(cost_tol)

        # BCJR / SOVA 관련 파라미터
        self.kappa_bcjr = float(kappa_bcjr)
        self.damp_L = float(damp_L)
        self.damp_beta = float(damp_beta)
        self.damp_zeta = float(damp_zeta)

        # similarity (bigger is better)
        mx = np.max(self.D)
        self.s = mx - self.D

        # trellis 크기
        self.T = self.N
        self.M = 1 << self.N

        # 단일 forward trellis ψ (with-β)
        self.psi = np.full((self.T + 1, self.M, self.N), NEG)
        self.backptr = np.full((self.T + 1, self.M, self.N, 2), -1, dtype=int)

        # backward trellis (with-β, closure 포함)
        self.bwd_wb = np.full((self.T + 1, self.M, self.N), NEG)

        # α_t(a): ψ_{t-1}로부터 계산
        self.alpha = np.full((self.T, self.N), NEG)

        # simplified messages
        self.gamma_tilde = np.zeros((self.N, self.T))
        self.omega_tilde = np.zeros((self.N, self.T))
        self.phi_tilde   = np.zeros((self.N, self.T))
        self.eta_tilde   = np.zeros((self.N, self.T))
        self.rho_tilde   = np.zeros((self.N, self.T))
        self.delta_tilde = np.zeros((self.N, self.T))

        # λ_t(i,a), ζ_t(i,a), β_t(a)
        self.lambda_ = np.zeros((self.T, self.N, self.N))
        self.zeta    = np.zeros((self.T, self.N, self.N))
        self.beta    = np.zeros((self.T, self.N))

        # damping 캐시
        self._L_prev    = [np.zeros((self.N, self.N)) for _ in range(self.T)]
        self._beta_prev = [np.zeros(self.N)            for _ in range(self.T)]
        self._zeta_prev = [np.zeros((self.N, self.N)) for _ in range(self.T)]

    # ===================== Public =====================
    def run(self):
        stable = 0
        last_cost = None
        best_route, best_cost = None, None

        for it in range(self.iters):
            # (1) forward trellis & α
            self._trellis_forward_and_alpha()

            # (2) backward trellis (primal with β + closure)
            self._trellis_backward_bcjr()

            # (3) BCJR-style SOVA LLR 계산 (T x N)
            llr_t = self._compute_sova_llr_from_bcjr()

            # (4) 메시지 업데이트
            self._update_phi_eta_rho()
            self._update_lambda_beta_zeta_delta(
                kappa=self.kappa_bcjr,
                llr_t=llr_t,
                damp_L=self.damp_L,
                damp_beta=self.damp_beta,
                damp_zeta=self.damp_zeta,
            )
            self._update_gamma_omega()

            if self.tiny_tiebreak:
                self.gamma_tilde += 1e-12 * self.rng.standard_normal(self.gamma_tilde.shape)

            # (5) route & cost
            route = self.estimate_route()
            cost = self._route_cost(route)

            if self.verbose:
                print(f"[{it+1:03d}] cost={cost:.12f} route={route}")

            # best primal 추적
            if best_cost is None or cost < best_cost:
                best_cost, best_route = cost, route

            # plateau early stop
            if last_cost is not None and abs(cost - last_cost) <= self.cost_tol:
                stable += 1
            else:
                stable = 0
            last_cost = cost

            if stable >= self.patience_no_cost_change:
                return best_route, best_cost

        return best_route, best_cost

    # ===================== Trellis (single ψ) & α =====================
    def _trellis_forward_and_alpha(self):
        """
        ψ_t(m_t, last) : with-β forward metric
        α_t(a)         : ψ_{t-1}로부터 계산 (현재 시점 β_t는 포함되지 않음)
        """
        self.psi.fill(NEG)
        self.backptr.fill(-1)
        self.alpha.fill(NEG)

        # t = 1
        # α_1(a) = s(depot, a)
        # ψ_1(m={a}, a) = s(depot, a) + β_1(a)
        for a in range(self.N):
            m = 1 << a
            gain = self.s[self.depot, a]
            self.alpha[0, a] = gain                    # α_1(a)
            self.psi[1, m, a] = gain + self.beta[0, a]  # ψ_1
            self.backptr[1, m, a] = (0, -1)             # 센티넬

        # t = 2..T
        full_mask = (1 << self.N) - 1
        for t in range(2, self.T + 1):
            # 1) α_t(a) 계산: ψ_{t-1}에서 오는 factor→A_t 메시지
            #    α_t(a) = max_{m_{t-1},last∈m_{t-1}, a∉m_{t-1}}
            #                 [ ψ_{t-1}(m_{t-1},last) + s(last,a) ]
            for a in range(self.N):
                best_alpha = NEG
                for mask in range(self.M):
                    if mask == 0 or mask.bit_count() != (t - 1):
                        continue
                    if mask & (1 << a):
                        continue  # 아직 방문 안 한 도시만
                    m = mask
                    while m:
                        last = (m & -m).bit_length() - 1
                        m ^= (1 << last)
                        cand = self.psi[t - 1, mask, last] + self.s[last, a]
                        if cand > best_alpha:
                            best_alpha = cand
                self.alpha[t - 1, a] = best_alpha

            # 2) ψ_t 전이 (with-β)
            for mask in range(self.M):
                if mask == 0 or mask.bit_count() != (t - 1):
                    continue
                for a in range(self.N):
                    if mask & (1 << a):
                        continue
                    new_mask = mask | (1 << a)
                    best = NEG
                    best_last = -1

                    m = mask
                    while m:
                        last = (m & -m).bit_length() - 1
                        m ^= (1 << last)
                        if last == a:
                            continue  # self-loop 금지

                        # ψ_{t-1} + s(last,a) + β_t(a)
                        cand = self.psi[t - 1, mask, last] + self.s[last, a] + self.beta[t - 1, a]
                        if cand > best:
                            best = cand
                            best_last = last

                    if best > self.psi[t, new_mask, a]:
                        self.psi[t, new_mask, a] = best
                        self.backptr[t, new_mask, a] = (mask, best_last)

    # ===================== Backward trellis (BCJR용) =====================
    def _trellis_backward_bcjr(self):
        """
        bwd_wb[t, mask, last]:
          - t 시점에 (mask,last) 상태에서 시작해서
          - 나머지 도시 방문 + depot으로 귀환까지의 최대 future metric.
        전이:
          - t < T:
              bwd[t,mask,last] = max_{a not in mask} { s[last,a] + β_{t+1}(a) + bwd[t+1, mask|{a}, a] }
            (코드에선 β_{t+1}(a)를 beta[t, a]로 저장)
          - t = T:
              full_mask 에 대해서만 closure: s[last, depot]
        """
        self.bwd_wb.fill(NEG)
        full_mask = (1 << self.N) - 1

        # t = T: full mask에서 depot으로 가는 closure
        t = self.T
        for last in range(self.N):
            mask = full_mask
            self.bwd_wb[t, mask, last] = self.s[last, self.depot]

        # t = T-1..1 역순
        for t in range(self.T - 1, 0, -1):
            for mask in range(self.M):
                if mask == 0 or mask.bit_count() != t:
                    continue
                for last in range(self.N):
                    if not (mask & (1 << last)):
                        continue

                    best = NEG
                    avail = (~mask) & full_mask
                    m = avail
                    while m:
                        a = (m & -m).bit_length() - 1
                        m ^= (1 << a)
                        new_mask = mask | (1 << a)
                        cand = self.s[last, a] + self.beta[t, a] + self.bwd_wb[t + 1, new_mask, a]
                        if cand > best:
                            best = cand
                    self.bwd_wb[t, mask, last] = best

    # ===================== SOVA-style LLR from BCJR =====================
    def _compute_sova_llr_from_bcjr(self):
        """
        llr[t, i] ≈
          max_{mask, last=i, |mask|=t+1} [ ψ[t+1, mask, i] + bwd[t+1, mask, i] ]
          - max_{mask, last≠i, |mask|=t+1} [ ψ[t+1, mask, last] + bwd[t+1, mask, last] ]
        여기서 내부 시간 인덱스(t+1)는 1..T 와 매칭, 외부 t는 0..T-1.
        """
        T, N, M = self.T, self.N, self.M
        llr = np.zeros((T, N), dtype=float)

        for t in range(1, T + 1):  # 내부 시간: 1..T
            for i in range(N):
                best_with = NEG
                best_without = NEG

                for mask in range(M):
                    if mask == 0 or mask.bit_count() != t:
                        continue

                    # last = i 인 state
                    val_i = self.psi[t, mask, i] + self.bwd_wb[t, mask, i]
                    if val_i > best_with:
                        best_with = val_i

                    # last ≠ i 인 state들 중 최고값
                    m2 = mask
                    while m2:
                        last = (m2 & -m2).bit_length() - 1
                        m2 ^= (1 << last)
                        if last == i:
                            continue
                        val = self.psi[t, mask, last] + self.bwd_wb[t, mask, last]
                        if val > best_without:
                            best_without = val

                if best_with <= NEG / 2 and best_without <= NEG / 2:
                    llr[t - 1, i] = 0.0
                else:
                    llr[t - 1, i] = best_with - best_without

        return llr

    # ===================== Messages =====================
    def _update_phi_eta_rho(self):
        # φ̃_it = -max_{i'≠i} γ̃_i't
        for t in range(self.T):
            col = self.gamma_tilde[:, t]
            for i in range(self.N):
                self.phi_tilde[i, t] = -np.max(np.delete(col, i)) if self.N > 1 else 0.0
        # η̃_it = -max_{t'≠t} ω̃_it'
        for i in range(self.N):
            row = self.omega_tilde[i, :]
            for t in range(self.T):
                self.eta_tilde[i, t] = -np.max(np.delete(row, t)) if self.T > 1 else 0.0
        # ρ̃_it
        self.rho_tilde = self.eta_tilde + self.phi_tilde

    def _update_lambda_beta_zeta_delta(self, kappa=0.0, llr_t=None,
                                       damp_L=0.5, damp_beta=0.5, damp_zeta=0.5):
        T, N = self.T, self.N

        for t in range(T):
            # (1) rhõ -> (rho0, rho1) 복원
            r = self.rho_tilde[:, t]  # shape (N,)
            rho0 = -r / N             # off-diagonal
            rho1 = rho0 + r           # diagonal

            # (2) L_new(i,a): a==i→rho1, else→rho0
            L_new = np.empty((N, N), float)
            for i in range(N):
                L_new[i, :] = rho0[i]
                L_new[i, i] = rho1[i]

            # (3) λ 이중 센터링
            L_new -= L_new.mean(axis=1, keepdims=True)
            L_new -= L_new.mean(axis=0, keepdims=True)

            # (4) λ damping
            L_prev = self._L_prev[t]
            L = damp_L * L_new + (1 - damp_L) * L_prev
            self.lambda_[t] = L

            # (5) β_t(a) = sum_i λ_{it}(a)
            beta_new = L.sum(axis=0)
            beta_new -= beta_new.mean()

            # (6) β damping
            beta_prev = self._beta_prev[t]
            beta_t = damp_beta * beta_new + (1 - damp_beta) * beta_prev
            self.beta[t, :] = beta_t

            # (7) ζ_it(a) = α_t(a) + β_t(a) - λ_it(a)
            a_t = self.alpha[t, :]
            Z_new = a_t[np.newaxis, :] + beta_t[np.newaxis, :] - L

            # (8) SOVA LLR 주입 (대각 성분)
            if kappa and llr_t is not None:
                for i in range(N):
                    Z_new[i, i] += kappa * llr_t[t, i]

            # (9) ζ damping
            Z_prev = self._zeta_prev[t]
            Z = damp_zeta * Z_new + (1 - damp_zeta) * Z_prev
            self.zeta[t] = Z

            # (10) δ̃_it = ζ_it(i) - max_{a≠i} ζ_it(a)
            for i in range(N):
                zi = Z[i, :]
                self.delta_tilde[i, t] = 0.0 if N == 1 else (zi[i] - np.max(np.delete(zi, i)))

        # 캐시 갱신
        self._L_prev    = [self.lambda_[t].copy() for t in range(T)]
        self._beta_prev = [self.beta[t].copy()    for t in range(T)]
        self._zeta_prev = [self.zeta[t].copy()    for t in range(T)]

    def _update_gamma_omega(self):
        gamma_new = self.eta_tilde + self.delta_tilde
        omega_new = self.phi_tilde + self.delta_tilde
        self.gamma_tilde = self.damp * gamma_new + (1 - self.damp) * self.gamma_tilde
        self.omega_tilde = self.damp * omega_new + (1 - self.damp) * self.omega_tilde

    # ===================== Decode =====================
    def estimate_route(self):
        full_mask = (1 << self.N) - 1
        best_val = NEG
        best_last = -1

        for last in range(self.N):
            base = self.psi[self.T, full_mask, last]
            if base <= NEG / 2:
                continue
            val = base + self.s[last, self.depot]  # closure
            if val > best_val:
                best_val = val
                best_last = last

        # backtrack
        if best_last < 0:
            # fallback: α 기반 greedy
            route_internal = [self.depot]
            used = set()
            for t in range(self.T):
                sc = self.alpha[t].copy()
                for u in used:
                    sc[u] = NEG
                if self.tiny_tiebreak:
                    sc += 1e-15 * np.arange(self.N)
                a = int(np.argmax(sc))
                used.add(a)
                route_internal.append(a)
            route_internal.append(self.depot)
        else:
            route_inner = []
            mask = full_mask
            last = best_last
            t = self.T
            while t > 0 and 0 <= last < self.N:
                route_inner.append(last)
                prev_mask, prev_last = self.backptr[t, mask, last]
                mask, last = prev_mask, prev_last
                t -= 1
            route_inner.reverse()
            route_internal = [self.depot] + route_inner + [self.depot]

        return [int(self.inv_perm[c]) for c in route_internal]

    def _route_cost(self, route):
        return float(sum(self.orig_D[route[k], route[k + 1]] for k in range(len(route) - 1)))