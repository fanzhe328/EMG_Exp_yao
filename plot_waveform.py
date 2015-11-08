# *-* coding=UTF-8 *-*

import numpy as np
import matplotlib.pyplot as plt
from data_load import load_dataset_yao


def main():
    trains = load_dataset_yao()
    print trains[0].shape[1]

    for i in range(len(trains)):
        for j in range(trains[i].shape[1]):
            channel = trains[i][:, j]
            times = np.linspace(0, len(channel) / 1024, len(channel))
            plt.figure(figsize=(20, 4))
            plt.plot(times, channel, label="trains_" + str(i) +
                     "_channel_" + str(j) + "_32s", color="red", linewidth=2)
            plt.xlabel("Times(s)")
            plt.ylabel("Volt")
            plt.title("trains_" + str(i) + "_channel_" + str(j) + "_32s")
            plt.xlim(0, len(channel) / 1024)
            plt.savefig("result/figure/trains_" + str(i) +
                        "_channel_" + str(j) + "_32s", dpi=120)
            plt.close()

    # channel1 = trains[0][:,0]
    # channel2 = trains[0][:2048,1]
    # # channel = np.reshape(channel, (1,-1))
    # # print len(channel), type(channel), channel.shape

    # times = np.linspace(0, len(channel1)/1024, len(channel1))

    # plt.figure(figsize=(20, 4))
    # # plt.plot(times, channel1, label="channel_1", color="red", linewidth=2)
    # plt.plot(times, channel2, label="channel_2", color="blue", linewidth=2)
    # plt.xlabel("Times(2)")
    # plt.ylabel("Volt")

    # plt.title("channel 1")
    # plt.xlim(0,2)
    # # plt.ylim(-15, 30)
    # plt.legend()
    # plt.show()
    # plt.savefig("channel_2", dpi=120)
    # print times.shape

if __name__ == '__main__':
    main()
