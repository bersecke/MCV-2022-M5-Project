import json
import matplotlib.pyplot as plt


f = open('./output/metrics_2cls.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)

losses = []
loss_cls = []
loss_box_reg = []
iters = []
for line in data:
    losses.append(line['total_loss'])   
    loss_cls.append(line['loss_cls'])   
    loss_box_reg.append(line['loss_box_reg'])   
    iters.append(line['iteration'])   

plt.plot(iters, losses, label='Total loss')
plt.plot(iters, loss_cls, label='Classification loss')
plt.plot(iters, loss_box_reg, label='Location loss')
plt.legend()
plt.savefig('2_classes_chart.png')