

hypo_plot_map = {0.25: 0.823254508455547, 0.5: 0.8433428060204188, 0.6: 0.8530456518879095, 0.1: 0.8545199391883991, 0.2: 0.834050391421617, 0.4: 0.8212788351898956, 0.15: 0.8403194848616184, 0.8: 0.9063930544593528, 0.3: 0.8208706112322381, 0.05: 0.8081320629293153, 0.9: 0.9230931088900578, 0.85: 0.9180224403927069, 0.75: 0.9003787878787879, 0.45: 0.8351928045163142, 0.7: 0.8714486145212206, 0.35: 0.8007934580195935, 0.95: 0.940273396424816, 0.55: 0.8189429373246024, 0.65: 0.8405892964521948}
unsuper_plot_map = {0.25: 0.86344817907515259, 0.5: 0.82254499526365643, 0.6: 0.85922904880936724, 0.1: 0.83557478657466966, 0.2: 0.87474508256035788, 0.4: 0.85738093149723704, 0.15: 0.84966875116091878, 0.8: 0.92265193370165743, 0.3: 0.86061198406134876, 0.05: 0.73725902947041877, 0.9: 0.92635455023671753, 0.85: 0.91479663394109401, 0.75: 0.90340909090909094, 0.45: 0.8492010333939336, 0.7: 0.87232549982462293, 0.35: 0.84875718565298353, 0.95: 0.917981072555205, 0.55: 0.86260523854069227, 0.65: 0.86725796752856288}

import matplotlib.pyplot as plt
import numpy as np
import operator

def create_plots():
    fig = plt.figure()
    #plt.scatter(hypo_plot_map.keys(), hypo_plot_map.values())
    percentage_hypo = []
    accuracy_hypo = []
    percentage_unsupervised = []
    accuracy_unsupervised = []
    for i in sorted(hypo_plot_map.keys()):
        percentage_hypo.append(i)
        accuracy_hypo.append(hypo_plot_map[i])

    for i in sorted(unsuper_plot_map.keys()):
        percentage_unsupervised.append(i)
        accuracy_unsupervised.append(unsuper_plot_map[i])
    plt.plot(percentage_hypo, accuracy_hypo, '-r*', label='Hypothesis')
    plt.plot(percentage_unsupervised, accuracy_unsupervised, '-g^', label='Supervised')
    ax = fig.gca()
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_yticks(np.arange(0.6,1,0.05))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.6, 1.0])
    plt.grid()
    plt.legend(loc='best')
    plt.title('Percent of Training v/s Accuracy')
    plt.xlabel('Percent of Data')
    plt.ylabel('Accuracy')
    plt.show()
    pass

if __name__ == '__main__':
    create_plots()