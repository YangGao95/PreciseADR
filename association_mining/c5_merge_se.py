import pickle as pk

if __name__ == '__main__':
    gender_SE, age_SE = pk.load(open("./gender_SE_set.pk", "rb")), pk.load(open("./age_SE_set.pk", "rb"))
    print(len(gender_SE), len(age_SE))
    gender_SE = gender_SE.union(age_SE)
    pk.dump(gender_SE, open("./all_SE_set.pk", "wb"))
    print(len(gender_SE))
