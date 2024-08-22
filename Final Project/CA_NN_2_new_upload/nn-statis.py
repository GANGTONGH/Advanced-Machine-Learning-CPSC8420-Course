#!/usr/bin/env python3
import sys
import numpy as np
import subprocess
import math

SVDSolver = "./SVD/bin/SVDSolver.linux"

def usage():
    print("usage: nn-statis.py CA_NN_Infor data")
    sys.exit(1)

# Decide format of well potential from CA_NN_info
# based on possible SSE
def read_ca_nn_info(in_filename):
    map_list = []
    range_list = []
    steps_list = []
    with open(in_filename, 'r') as in_file:
        for line in in_file:
            if line[0] == "#":
                continue
            parts = line.split()
            if len(parts) > 1:
                map_list.append(parts[0])
                steps_list.append(len(parts) - 1)
                range_list.append([float(x) for x in parts[1:]])
        in_file.close()
    return map_list, range_list, steps_list

def read_ca_nn_data(in_filename, cut, map_list, steps_list):
    # Initialize
    statis = np.zeros((len(map_list), len(map_list), len(map_list), max(steps_list) // 2), dtype=int)
    res_nrec = np.zeros(len(map_list), dtype=int)

    with open(in_filename, 'r') as in_file:
        for line in in_file:
            parts = line.split()
            i0 = map_list.index(parts[0])
            i1 = map_list.index(parts[1])
            i2 = map_list.index(parts[2])
            value = float(parts[3])
            
            irange = -1
            for i in range(steps_list[i1] // 2 - 1):
                if value < cut[i1][i]:
                    irange = i
                    break
            if irange < 0 and value < 7.66:
                irange = steps_list[i1] // 2 - 1

            if irange >= 0:
                statis[i0][i1][i2][irange] += 1
                res_nrec[i1] += 1
    return statis, res_nrec

def perform_bayesion_statistics(statis, res_nrec, steps_list, map_list):
    for c in range(len(map_list)):
        states = steps_list[c] // 2
        print(states)
        if res_nrec[c] > 0:
            PN_S = np.zeros((20, 20, states))
            # PN_S = np.zeros((20, 20, 3))
            for p in range(len(map_list)):
                for n in range(len(map_list)):
                    tmp_sum = np.sum(statis[p][c][n])
                    if tmp_sum == 0:
                        tmp_ps = np.zeros(states)
                        tmp_ns = np.zeros(states)
                        nnn = 0
                        for nn in range(len(map_list)):
                            if nn != n:
                                ss_sum = np.sum(statis[p][c][nn])
                                if ss_sum > 0:
                                    nnn += 1
                                    tmp_ns += statis[p][c][nn] / ss_sum
                        if nnn > 0:
                            tmp_ns /= nnn
                        nnn = 0
                        for pp in range(len(map_list)):
                            if pp != p:
                                ss_sum = np.sum(statis[pp][c][n])
                                if ss_sum > 0:
                                    nnn += 1
                                    tmp_ps += statis[pp][c][n] / ss_sum
                        if nnn > 0:
                            tmp_ps /= nnn
                        tmp_sum = np.sum(tmp_ns * tmp_ps)
                        if tmp_sum > 0:
                            PN_S[p][n] = tmp_ns * tmp_ps
                            PN_S[p][n] /= np.sum(PN_S[p][n])
                    else:
                        # PN_S[p][n] = statis[p][c][n] / tmp_sum
                        PN_S[p][n] = statis[p][c][n][:states] / tmp_sum
                        PN_S[p][n] = np.clip(PN_S[p][n], 0.01, 0.98)
            
            # Create MATRIX/VECTOR for SVD
            M = 20 * 20 * states
            N = 20 * states * 2
            N_HALF = 20 * states
            A = np.zeros((M, N))
            B = np.zeros(M)
            index = 0

            for i in range(20):
                for j in range(20):
                    for k in range(states):
                        B[index] = PN_S[i][j][k]
                        p_s = i * states + k
                        n_s = j * states + k + N_HALF
                        A[index][p_s] = 1
                        A[index][n_s] = 1
                        index += 1

            Amatrix = f"{map_list[c]}.MATRIX"
            Bvector = f"{map_list[c]}.VECTOR"
            Xsol = f"{map_list[c]}.SOL"

            np.savetxt(Amatrix, A, fmt='%d', delimiter=' ')
            np.savetxt(Bvector, B, fmt='%f')

            # Run the SVD solver
            subprocess.run([SVDSolver, str(M), str(N), Amatrix, Bvector, "0.00001"], check=True)

            # Read the SVD data
            PSI_AA = np.zeros((20, states))
            PHI_AA = np.zeros((20, states))
            with open(Xsol, 'r') as f:
                next(f)  # Skip the first line
                iline = 0
                for line in f:
                    parts = line.split()
                    value = math.exp(float(parts[1]))
                    if iline >= 20 * states:
                        psi_line = iline - 20 * states
                        psi = psi_line // states
                        iaa = psi_line % states
                        PSI_AA[psi][iaa] = value
                    else:
                        phi_line = iline
                        phi = phi_line // states
                        iaa = phi_line % states
                        PHI_AA[phi][iaa] = value
                    iline += 1

            # Sanity check
            if np.any(np.isnan(PSI_AA)) or np.any(np.isnan(PHI_AA)):
                print("ERROR")
                sys.exit(1)

            # Obtain Xi and expectation values
            K = 0.5
            alpha = np.zeros((20, 20, states))
            exp = np.zeros((20, 20, states))
            exp_counts = np.zeros((20, 20, states))

            for i in range(20):
                for j in range(20):
                    for iaa in range(states):
                        alpha[i][j][iaa] = PSI_AA[i][iaa] * PHI_AA[j][iaa]
                    Y0 = np.sum(statis[i][c][j])
                    A0 = np.sum(alpha[i][j])
                    OCCUP = np.clip(Y0 * K, states, states * 5)
                    scale = OCCUP / A0
                    alpha[i][j] *= scale
                    exp[i][j] = (alpha[i][j] + statis[i][c][j][:states] + 1) / (OCCUP + Y0 + states)
                    Y0 = max(Y0, 1)
                    exp_counts[i][j] = Y0 * (alpha[i][j] + statis[i][c][j][:states] + 1) / (OCCUP + Y0 + states)

            # Output posterior data
            for i in range(20):
                for j in range(20):
                    for k in range(states):
                        print(f"{map_list[i]} {map_list[c]} {map_list[j]} {k:4d}: {statis[i][c][j][k]:4d} {exp_counts[i][j][k]:8.3f} {exp[i][j][k]:8.3f}")

def main():
    if len(sys.argv) < 3:
        usage()

    ca_nn_info_file = sys.argv[1]
    ca_nn_data_file = sys.argv[2]

    map_list, range_list, steps_list = read_ca_nn_info(ca_nn_info_file)

    cut = np.zeros((len(map_list), max(steps_list) // 2 - 1))
    for i in range(len(map_list)):
        for j in range(steps_list[i] // 2 - 1):
            cut[i][j] = (range_list[i][2 * j + 1] + range_list[i][2 * j + 2]) / 2

    statis, res_nrec = read_ca_nn_data(ca_nn_data_file, cut, map_list, steps_list)
    perform_bayesion_statistics(statis, res_nrec, steps_list, map_list)

if __name__ == "__main__":
    main()
