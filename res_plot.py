import matplotlib.pyplot as plt
import pickle



with open(
        "generated_outputs_nsample1.pk", "rb"
) as f:
    all_generated_samples,all_target,_,_,_=pickle.load(f)

B,  L, _ = all_target.shape

generated1=all_generated_samples.mean(dim=1).to('cpu').numpy().reshape(B,L)

targ1=all_target.to('cpu').numpy().reshape(B,L)

for j in range(20):
    plt.plot(generated1[j], '-*')
    plt.plot(targ1[j])
    plt.show()

