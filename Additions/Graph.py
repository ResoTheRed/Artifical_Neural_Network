from matplotlib import pyplot as plt
x = [i+1 for i in range(10)]
y1 = [0.22219,0.22175,0.22219,0.22211,0.22164,0.22239,0.22219,0.22208,0.22219,0.22239]
y2 = [0.66656,0.66524,0.66656,0.66634,0.66491,0.66656,0.66656,0.66554,0.66656,0.66666]
plt.plot(x,y1,label="Training")
plt.plot(x,y2,label="Testing")
plt.xlabel("Folds")
plt.ylabel("MSE")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
plt.show()
