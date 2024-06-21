import pickle as pk

if __name__ == '__main__':
    gender_SE, age_SE = pk.load(open("./gender_SE_set.pk", "rb")), pk.load(open("./age_SE_set.pk", "rb"))
    gender_SE_pre, age_SE_pre = pk.load(open("../gender_SE_set.pk", "rb")), pk.load(open("../age_SE_set.pk", "rb"))
    print(len(gender_SE_pre.intersection(gender_SE)))
    print(len(gender_SE_pre.difference(gender_SE)))
    print(gender_SE.difference(gender_SE_pre))

    print(len(age_SE_pre.intersection(age_SE)))
    print(age_SE_pre.difference(age_SE))
    print(age_SE.difference(age_SE_pre))