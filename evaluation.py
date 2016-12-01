import matplotlib

from seiz.learn import SeizureModel

matplotlib.use("Agg")

model = SeizureModel("../data/features/train", "../data/features/test", patients=[1, 2, 3])


def production(sub_name):
    model.train()
    sub_df = model.create_submission()
    print(model.get_info())
    sub_df.to_csv(sub_name, columns=['File', 'Class'], index=False)

# cv = model.do_cross_validation()
# print(cv)

production('submission_tmp.csv')

