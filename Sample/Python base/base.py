from math import *
print(str(1))
print(max(2,3,88,100))
print(min(2,3,88,100))
print(round(4.4)) #四捨五入
print(floor(4.6))
print(sqrt(36))

scores = [33,34,35,36]
print(scores[:2]) #start~2
print(scores[1:]) #1~end
scores.extend([0,1])
print(scores)
scores.append(30)
print(scores)
scores.insert(2, 100)
print(scores)
scores.pop()
print(scores) #移除最後一位
scores.sort()
print(scores)
scores.reverse()
print(scores)
print(scores.index(34))
print(scores.count(30))
scores.clear()
print(scores)

#tuple 差別:無法修改
scores = (90,80,74,60,92)
print(scores[3])
print(len(scores))

#dictionary key value
dic = {"a":"apple", "b":"banana"}
dic1 = {0:"apple"}
print(dic["a"])
print(dic1[0])


##class
class Phone:
    def __init__(self,os,number,is_waterproof):
        self.os=os
        self.number=number
        self.is_waterproof=is_waterproof


phone1 = Phone("ios", "0911111111", True)

print(phone1.os)