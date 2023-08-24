def reverse_array(arr):
    # Return the reversed version of the array
    return arr[::-1]


# Test the function
arr = [
    [5.6, 14.45],
    [5.603331224463497, 14.297332144221324],
    [5.61331855667889, 14.144954900383196],
    [5.629942985191663, 13.99315832722982],
    [5.653172864457272, 13.842231378165744],
    [5.6829639750802325, 13.69246135121664],
    [5.719259607988261, 13.544133342141176],
    [5.761990672381206, 13.397529701735044],
    [5.81107582724932, 13.252929498360158],
    [5.866421636210496, 13.110607986722185],
    [5.927922745371724, 12.970836083907551],
    [5.995462083876223, 12.83387985367738],
    [6.068911086754464, 12.7],
    [6.1481299396549005, 12.569451370786116],
    [6.232967844988528, 12.442482472771339],
    [6.3232633089806765, 12.319334998469477],
    [6.4188444490835765, 12.200243366097112],
    [6.5195293211645655, 12.08543427334519],
    [6.625126265847083, 11.975126265847084],
    [6.735434273345188, 11.869529321164565],
    [6.850243366097112, 11.768844449083577],
    [6.969334998469477, 11.673263308980676],
    [7.092482472771337, 11.582967844988529],
    [7.219451370786115, 11.4981299396549],
    [7.349999999999998, 11.418911086754465],
    [7.483879853677379, 11.345462083876225],
    [7.620836083907553, 11.277922745371724],
    [7.760607986722183, 11.216421636210496],
    [7.90292949836016, 11.16107582724932],
    [8.04752970173504, 11.111990672381205],
    [8.194133342141178, 11.069259607988261],
    [8.34246135121664, 11.032963975080232],
    [8.492231378165744, 11.003172864457271],
    [8.64315832722982, 10.979942985191663],
    [8.794954900383196, 10.963318556678889],
    [8.947332144221324, 10.953331224463497],
    [9.1, 10.95],
]
reversed_arr = reverse_array(arr)

for p in reversed_arr:
    print("[ ", p[0], " , ", p[1], " ],")
