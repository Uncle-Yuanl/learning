import itertools as it


# m = it.groupby('aaabbccccaaaa')
# for k, v in m:
#     print(k, list(v))


# m = it.groupby([1,2,2,3,3,3,'A','B','B','C','C','C'])
# for k, v in m:
#     print(k, list(v))


# m = it.groupby(zip([1,2,2,3,3,3], ['A','B','B','C','C','C']))
# for k, v in m:
#     print(k, list(v))


# lambda表示分组依据
m = it.groupby('aaabbccccaaaa', lambda x: x == 'a')
for k, v in m:
    print(k, list(v))


# 会将满足/不满足条件的优先放在一个组里
m = it.groupby('66677711223344555666666', lambda x: int(x) > 3)
for k, v in m:
    print(k, list(v))

# tuple
a = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
b = ['a', 'a', 'b', 'a', 'a', 'b', 'a', 'a', 'b', 'b']
m = it.groupby(zip(a, b))
for k, v in m:
    print(k, list(v))

# 不指定就是by所有字段，指定了就是by指定字段
m = it.groupby(zip(a, b), lambda x: x[0])
for k, v in m:
    print(k, list(v))
