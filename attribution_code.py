import openpyxl as openpyxl
import pandas as pd
import numpy as np
import xlwt as xlwt


pd.options.display.max_columns = 35

# ----- import data and simple processing-----
df = pd.read_csv('subscribers_aa.csv')
channel_list = ['bing', 'display', 'facebook', 'search', 'youtube']
df_last = df.loc[df['attribution_technical'].isin(channel_list)]
df_tier = pd.read_csv('subid_tier_spend.csv')


channel_spend = pd.read_csv('channel_spend_undergraduate.csv')
# print(df_last.head())
# print(df_first.head())
print(1)
# ----- total spending for all eight tier experiments
total_spend = {'bing': 10800, 'display': 366, 'facebook': 113500, 'search': 222500, 'youtube': 8730}

# ----- calculate revenue for each channel -----
channels_rev = df_last.groupby(['attribution_technical'])['clv'].sum().to_dict()
# print(channels_rev)

# ----- see if the channel is long-term sustainable -----
sustainability = {}
for channel in channels_rev.keys():
    if total_spend.get(channel) > channels_rev.get(channel):
        sustainability[channel] = 'unsustainable'
    else:
        sustainability[channel] = 'sustainable'

# ----- number of occurrences generated for each channel
df_number_customers = df_last['attribution_technical'].value_counts()
number_customers = df_number_customers.to_dict()
# print(number_customers)

# ----- calculate average CAC(for one customer)-----
average_cac = {}
for item in number_customers.keys():
    average_cac[item] = total_spend[item] / number_customers[item]
print(2)
# print(average_cac)

# ----- export results -----
results = {}
data_list = []
for keys in total_spend.keys():
    data_list.append(total_spend[keys])
    data_list.append(channels_rev[keys])
    data_list.append(sustainability[keys])
    data_list.append(number_customers[keys])
    data_list.append(average_cac[keys])
    results[keys] = data_list
    data_list = []

for_export = pd.DataFrame(data=results).T
for_export.columns = ['total_cac', 'clv', 'sustainability', 'number_of_customers', 'average_cac']
print(3)
for_export.to_excel("average_cac.xlsx")
print(4)

# print(for_export)


# ----- calculate marginal cac -----

# --- define function ---
def calc_marginal_CAC(marginal_conversions, marginal_spend):
    marginal_CAC = marginal_spend / marginal_conversions
    return [marginal_conversions, marginal_spend, marginal_CAC]


# --- merge two data frames ---
df_tiers = df_tier.merge(df[['attribution_technical', 'attribution_survey', 'subid']].drop_duplicates(),
                         how='left', on='subid')
# print(df_tiers.head())

# --- count number of customers via each tier (marginal numbers of customers)---
# - bing -
df_bing = df_tiers[df_tiers['attribution_technical'] == 'bing']
bing_refer = df_bing['tier'].value_counts().to_dict()
# - display -
df_dis = df_tiers[df_tiers['attribution_technical'] == 'display']
dis_refer = df_dis['tier'].value_counts().to_dict()
# - facebook -
df_fb = df_tiers[df_tiers['attribution_technical'] == 'facebook']
fb_refer = df_fb['tier'].value_counts().to_dict()
# - search -
df_search = df_tiers[df_tiers['attribution_technical'] == 'search']
search_refer = df_search['tier'].value_counts().to_dict()
# - youtube -
df_yt = df_tiers[df_tiers['attribution_technical'] == 'youtube']
yt_refer = df_yt['tier'].value_counts().to_dict()

# --- calculate marginal spending ---
spending_dict = {1: {'bing': 300, 'display': 12, 'facebook': 9000, 'search': 13000, 'youtube': 90},
                 2: {'bing': 400, 'display': 13, 'facebook': 10500, 'search': 18500, 'youtube': 100},
                 3: {'bing': 900, 'display': 19, 'facebook': 11000, 'search': 19000, 'youtube': 130},
                 4: {'bing': 1000, 'display': 20, 'facebook': 13000, 'search': 24000, 'youtube': 180},
                 5: {'bing': 1100, 'display': 29, 'facebook': 14000, 'search': 25000, 'youtube': 550},
                 6: {'bing': 1300, 'display': 31, 'facebook': 16000, 'search': 38000, 'youtube': 900},
                 7: {'bing': 2100, 'display': 94, 'facebook': 17000, 'search': 41000, 'youtube': 2420},
                 8: {'bing': 3700, 'display': 148, 'facebook': 23000, 'search': 44000, 'youtube': 4360}}

bing_spend = {1: 300}
dis_spend = {1: 12}
fb_spend = {1: 9000}
search_spend = {1: 13000}
yt_spend = {1: 90}

for val in range(7):
    tier_spend = spending_dict[val + 1]
    tier_spend_next = spending_dict[val + 2]
    for item in tier_spend:
        if item == 'bing':
            bing_spend[val + 2] = tier_spend_next[item] - tier_spend[item]
        elif item == 'display':
            dis_spend[val + 2] = tier_spend_next[item] - tier_spend[item]
        elif item == 'facebook':
            fb_spend[val + 2] = tier_spend_next[item] - tier_spend[item]
        elif item == 'search':
            search_spend[val + 2] = tier_spend_next[item] - tier_spend[item]
        else:
            yt_spend[val + 2] = tier_spend_next[item] - tier_spend[item]

# print('bing_spend', bing_spend)
# print('bing_refer', bing_refer)
# --- calculate marginal CAC ---
conversions_summary = {'bing': bing_refer, 'display': dis_refer, 'facebook': fb_refer,
                       'search': search_refer, 'youtube': yt_refer}
spend_summary = {'bing': bing_spend, 'display': dis_spend, 'facebook': fb_spend,
                 'search': search_spend, 'youtube': yt_spend}

for items in channel_list:
    marginal_summary = {}
    for sn in range(8):
        source_conversion = conversions_summary[items]
        # print(source_conversion)
        source_spend = spend_summary[items]
        marginal_summary[sn + 1] = calc_marginal_CAC(source_conversion[sn + 1], source_spend[sn + 1])
        # print(marginal_summary)
    summary = pd.DataFrame(data=marginal_summary).T
    summary.columns = ['marginal_conversions', 'marginal_spend', 'marginal_CAC']
    # print(summary.head(8))
    summary.to_excel(items + '_marginal.xls')
