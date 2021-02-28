import matplotlib.pyplot as plt
import numpy as np
import pickle 


with open('pickles_for_plot/newcollege_first_vlad.pkl','rb') as f:  
    pr1 = pickle.load(f)

with open('pickles_for_plot/newcollege_first_seq.pkl','rb') as f:  
    pr2 = pickle.load(f)


plt.plot(pr1[:, 1], pr1[:, 0], '--')
plt.plot(pr2[:, 1], pr2[:, 0], '-.')

plt.title('PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1.05, 0, 1.05])
plt.grid()
plt.legend(['vlad', 'seq'], loc="upper right")
plt.show()



