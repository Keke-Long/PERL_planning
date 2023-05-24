import pandas as pd

platoon_num = 10

platoon_nums = [3,10]  # Example platoon numbers
dfs = []

for platoon_num in platoon_nums:
    df = pd.read_csv("ASta_platoon{}.csv".format(platoon_num), skiprows=range(5))

    a = '1'
    b = '2'
    new_df1 = df.loc[:, ['Time', 'E'+a, 'N'+a, 'E'+b, 'N'+b, 'Speed'+a, 'Speed'+b, 'IVS'+a]]
    time_diff = new_df1['Time'].diff()
    new_df1['A2'] = new_df1['Speed'+b].diff() / time_diff

    def add_data(a,b):
        new_df2 = df.loc[:, ['Time', 'E'+a, 'N'+a, 'E'+b, 'N'+b, 'Speed'+a, 'Speed'+b, 'IVS'+a]]
        new_df2 = new_df2.rename(columns={'Speed'+a: 'Speed1'})
        new_df2 = new_df2.rename(columns={'Speed'+b: 'Speed2'})

        new_df2 = new_df2.rename(columns={'E' + a: 'E1'})
        new_df2 = new_df2.rename(columns={'N' + a: 'N1'})
        new_df2 = new_df2.rename(columns={'E' + b: 'E2'})
        new_df2 = new_df2.rename(columns={'N' + b: 'N2'})

        new_df2 = new_df2.rename(columns={'IVS'+a: 'IVS1'})
        new_df2['A2'] = new_df2['Speed2'].diff() / time_diff
        return new_df2

    new_df2 = add_data('2', '3')
    new_df3 = add_data('3', '4')
    new_df4 = add_data('4', '5')

    df = pd.concat([new_df1, new_df2, new_df3, new_df4])
    df = df[df['A2'].notnull()]
    # df.to_csv("../Data/ASta_platoon{}_new.csv".format(platoon_num), index=False)
    dfs.append(df)

combined_df = pd.concat(dfs)
combined_df.to_csv("../Data/ASta_platoon20_new.csv", index=False)