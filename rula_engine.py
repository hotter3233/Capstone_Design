class RULAEngine:
    def __init__(self):
        # Table A: 상완(6) x 전완(3) x 손목(4) x 손목 비틀림(2)
        self.table_A = {
            1: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:2, 2:3}, 4: {1:3, 2:3}},
                 2: {1: {1:2, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:3}},
                 3: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:3}, 4: {1:4, 2:4}} },
            2: { 1: {1: {1:2, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:3}},
                 2: {1: {1:2, 2:2}, 2: {1:3, 2:3}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
                 3: {1: {1:3, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}} },
            3: { 1: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
                 2: {1: {1:3, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
                 3: {1: {1:4, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:4}, 4: {1:5, 2:5}} },
            4: { 1: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
                 2: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
                 3: {1: {1:4, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}} },
            5: { 1: {1: {1:4, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
                 2: {1: {1:4, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
                 3: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:5, 2:5}, 4: {1:6, 2:6}} },
            6: { 1: {1: {1:5, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
                 2: {1: {1:5, 2:6}, 2: {1:5, 2:6}, 3: {1:6, 2:6}, 4: {1:7, 2:7}},
                 3: {1: {1:6, 2:6}, 2: {1:6, 2:7}, 3: {1:7, 2:7}, 4: {1:7, 2:8}} }
        }
        # Table B: 목(6) x 몸통(6) x 다리(2)
        self.table_B = {
            1: { 1: {1:1, 2:3}, 2: {1:2, 2:3}, 3: {1:3, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:6}, 6: {1:7, 2:7} },
            2: { 1: {1:2, 2:3}, 2: {1:2, 2:3}, 3: {1:4, 2:5}, 4: {1:5, 2:5}, 5: {1:6, 2:7}, 6: {1:7, 2:7} },
            3: { 1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:6}, 5: {1:6, 2:7}, 6: {1:7, 2:7} },
            4: { 1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:5, 2:6}, 4: {1:6, 2:7}, 5: {1:7, 2:7}, 6: {1:7, 2:8} },
            5: { 1: {1:7, 2:7}, 2: {1:7, 2:7}, 3: {1:7, 2:7}, 4: {1:7, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8} },
            6: { 1: {1:8, 2:8}, 2: {1:8, 2:8}, 3: {1:8, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:9} }
        }
        # Table C: Score A(1-8) x Score B(1-7)
        self.table_C = [
            [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5],
            [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
            [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7],
            [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]
        ]

    def get_upper_arm_score(self, angle, is_abducted_or_raised=False):
        a = abs(angle)
        score = 1 if a <= 20 else 2 if a <= 45 else 3 if a <= 90 else 4
        if is_abducted_or_raised: score += 1
        return min(score, 6)

    def get_lower_arm_score(self, angle, is_across_midline=False):
        score = 1 if 60 <= angle <= 100 else 2
        if is_across_midline: score += 1
        return min(score, 3)

    def get_wrist_score(self, angle, is_deviated=False):
        a = abs(angle)
        score = 1 if a <= 5 else 2 if a <= 15 else 3
        # RULA 기준 편위(구부러짐)나 극단적 꺾임 반영
        if is_deviated or a > 45: score += 1 
        return min(score, 4)

    def get_neck_score(self, angle, is_twisted_or_bent=False):
        a = abs(angle)
        score = 1 if a <= 10 else 2 if a <= 20 else 3 if a <= 45 else 4
        if is_twisted_or_bent: score += 1
        return min(score, 6)

    def get_trunk_score(self, angle, is_twisted_or_bent=False):
        a = abs(angle)
        score = 1 if a <= 10 else 2 if a <= 20 else 3 if a <= 60 else 4
        if is_twisted_or_bent: score += 1
        return min(score, 6)

    def calculate_final_score(self, u_a, l_a, w, w_t, n, t, l, m_a=0, f_a=0, m_b=0, f_b=0):
        try:
            score_a = self.table_A[u_a][l_a][w][w_t] + m_a + f_a
            score_b = self.table_B[n][t][l] + m_b + f_b
            
            # 🔴 버그 6 해결: 점수가 낮아져 인덱스가 음수가 되는 것을 방지 (max(0, ...) 추가)
            idx_a = max(0, min(int(score_a), 8) - 1)
            idx_b = max(0, min(int(score_b), 7) - 1)
            final = self.table_C[idx_a][idx_b]
            
            levels = {
                (1, 2): "수용 가능 (자세 양호)",
                (3, 4): "추가 조사 필요 (변경 고려)",
                (5, 6): "조만간 자세 개선 필요",
                (7, 8): "즉각적인 개선 필요"
            }
            action = next((v for k, v in levels.items() if final in k), "심각한 위험")
            return final, action
        except Exception as e:
            return 1, f"오류: {str(e)}"