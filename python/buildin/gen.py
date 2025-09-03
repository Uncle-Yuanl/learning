# def gen(n):
#     for i in range(n):
#         yield i


# # g = gen(10)
# # for _ in g:
# #     print(_)


# def make_gen(ns=[2,3]):
#     for n in ns:
#         g = gen(n)
        
#         yield g


# for g in make_gen():
#     print("++++++++++++")
#     for _ in g:
#         print(_)



stock = {
    "apple": 5
}

# numapp = stock.get("apple", 0)
# if numapp >= 4:
if (numapp := stock.get("apple", 0)) >= 4:
    print(f"Make juice with {numapp} apple")
else:
    print("Not OK")