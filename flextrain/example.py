import torch
from tqdm import tqdm


# class MyModel(torch.nn.Module):
#     def __init__(self, hidden_size, num_layers):
#         super(MyModel, self).__init__()

#         self.pre_embedding = torch.nn.Linear(hidden_size, hidden_size)
#         self.layers = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
#         self.post_embedding = torch.nn.Linear(hidden_size, 1)

#     def pre_process(self, x):
#         x = self.pre_embedding(x)
#         x = torch.relu(x)
#         return x

#     def post_process(self, x):
#         x = self.post_embedding(x)
#         x = torch.sigmoid(x)
#         return x

#     def custom_forward(self, x, layer_indices):
#         for i in layer_indices:
#             x = self.layers[i](x)
#             x = torch.relu(x)
#         return x

#     def forward(self, x):
#         x = self.pre_process(x)
#         x = self.custom_forward(x, list(range(len(self.layers))))
#         x = self.post_process(x)
#         return x


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size, *args):
        super(MyModel, self).__init__()

        self.linear = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


def target(x):
    feat1 = x[:, 0]
    return (feat1 > 0) * 1.0


torch.set_default_device("cuda:5")


x_train = torch.randn(64 * 1024, 1)
x_test = torch.randn(1024, 1)

y_train = target(x_train).reshape(-1, 1)
y_test = target(x_test).reshape(-1, 1)


model = MyModel(1, 6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
data_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=1024)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for x, y in data_loader:
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print("Loss:", loss.item())
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    loss = loss_fn(y_pred, y_test)
print("Loss:", loss.item())
