x = []
y = []
for i in range(10):
    x.append(i)
for j in range(5):
    y.append(j)
print(len(y))


def get_index(x_index, y_index):
    if x_index < 0:
        x_index = len(x) - x_index
    elif x_index >= len(x):
        x_index %= len(x)
    if y_index < 0:
        y_index = len(y) - y_index
    elif y_index >= len(y):
        y_index %= len(y)
    return x_index * len(y) + y_index


def get_xy(index):
    x_index = index // len(y)
    y_index = index - x_index * len(y)
    return x_index, y_index


z = []
for i in range(len(x)):
    for j in range(len(y)):
        z.append((x[i], y[j]))
print(z)

test = 34
print(z[test])
print(get_xy(test))
