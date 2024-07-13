import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = None
    
    # TODO

    if voting_rule == 'plurality':
        mostPreferred = np.array([row[0] for row in profiles])
        ElemFreq = np.bincount(mostPreferred)
        maxFreq = np.max(ElemFreq)
        for i in range(len(ElemFreq)):
            if ElemFreq[i] == maxFreq:
                return i
            
    elif voting_rule == 'borda':
        num_candidates = profiles.shape[1]
        num_voters = profiles.shape[0]
        candidate_scores = np.zeros(num_candidates)
        for i in range(num_voters):
            for j in range(num_candidates):
                candidate_scores[profiles[i][j] - 1] += num_candidates - j - 1
        maxScore = np.max(candidate_scores)
        for i in range(num_candidates):
            if (candidate_scores[i] == maxScore):
                return i+1

    elif voting_rule == 'stv':
        num_candidates = profiles.shape[1]
        eliminated_candidates = set()
        for _ in range(num_candidates - 1):
            mostPreferred = np.array([row[0] for row in profiles])
            ElemFreq = np.bincount(mostPreferred)
            minFreq = np.min(ElemFreq)
            worstCandidate = None
            for i in range(len(ElemFreq)):
                if ElemFreq[i] == minFreq and (i + 1 not in eliminated_candidates):
                    worstCandidate = i + 1
                    eliminated_candidates.add(worstCandidate)
                    break
            profiles = np.array([row[row != worstCandidate] for row in profiles])
        return profiles[0]


    elif voting_rule == 'copeland':
        num_candidates = profiles.shape[1]
        num_voters = profiles.shape[0]
        copeland_scores = np.zeros(num_candidates)
        
        for i in range(num_candidates):
            for j in range(i+1, num_candidates):
                num_i = 0
                num_j = 0
                for k in range(num_voters):
                    a_idx = np.where(profiles[k] == i+1)[0]
                    b_idx = np.where(profiles[k] == j+1)[0]
                    
                    if len(a_idx) > 0 and len(b_idx) > 0:
                        a = a_idx[0]
                        b = b_idx[0]
                        if a < b:
                            num_i += 1
                        elif a > b:
                            num_j += 1
                        else:
                            num_i += 0.5
                            num_j += 0.5
                
                if num_i > num_j:
                    copeland_scores[i] += 1
                elif num_i < num_j:
                    copeland_scores[j] += 1
                else:
                    copeland_scores[i] += 0.5
                    copeland_scores[j] += 0.5

        max_score = np.max(copeland_scores)
        winner_idx = np.argmin([i if score == max_score else float('inf') for i, score in enumerate(copeland_scores)])
        
        return winner_idx + 1

    # END TODO

    return winner_index


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = None

    # TODO
    
    ranks = np.array([np.where(row == winner)[0] for row in profiles])
    average_rank = np.mean(ranks) + 1

    # END TODO

    return average_rank


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''

    manipulable = None

    # TODO

    pass

    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(10, 20))
        num_candidates = np.random.choice(np.arange(4, 8))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')