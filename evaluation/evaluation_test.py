import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Data provided
#data = {
#    "Max depth": [1, 3, 3, 5, 1, 3, 99, 3],
#    "Beam width": [3, 1, 5, 3, 1, 3, 99, 2],
#    "Number of unique entailing triples": [9, 44, 46, 54, 9, 49, 74, 43],
#    "Total entailing triples selected for refinement": [2, 20, 19, 20, 2, 19, 30, 19],
#    "Number of direct or indirect refinements of a goal": [1, 11, 10, 11, 1, 10, 14, 10],
#    "Total leaf goals": [1, 5, 4, 5, 1, 4, 7, 4],
#    "Depth of the goal hierarchy": [1, 5, 5, 5, 1, 5, 5, 5],
    ##"Execution Time (s)": [130.91, 431.39, 632.79, 636.88, 68.76, 632.79, 2287.91, 453.83],
#}

data = {
    "Max depth": [1, 3, 3, 5, 1, 3, 99, 3],
    "Beam width": [3, 1, 5, 3, 1, 3, 99, 2],
    "Number of unique entailing triples": [9, 44, 46, 54, 9, 49, 75, 50],
    "Total entailing triples selected for refinement": [2, 20, 19, 20, 2, 19, 29, 18],
    "Number of direct or indirect refinements of a goal": [1, 11, 10, 11, 1, 10, 14, 10],
    "Number of leaf goals": [1, 5, 4, 5, 1, 4, 7, 4],
    "Depth of the goal hierarchy": [1, 5, 5, 5, 1, 5, 5, 5],
}

#data_time_execution = {
#    'GoalModel_MD1_BW3': [35.02, 48.27],
#    'GoalModel_MD3_BW1': [33.53, 56.82, 43.64, 38.48, 33.97, 27.47, 57.26, 41.50, 42.93, 22.37, 27.73, 28.06],
#    'GoalModel_MD3_BW5': [42.62, 88.29, 60.08, 64.03, 45.81, 45.53, 119.04, 40.33, 53.78, 26.77, 46.51],
#    'GoalModel_MD5_BW3': [41.71, 93.28, 51.72, 46.89, 38.07, 36.85 , 110.58, 74.97, 39.30, 25.86, 32.81, 44.84],
#    'GoalModel_MD1_BW1': [32.76, 36],
#    'GoalModel_MD3_BW3': [35.93, 90.32, 51.95, 45.74, 43.63, 38.03, 90.11, 73.63, 39.05, 36.64, 39.81],
#    'GoalModel_MD99_BW99': [71.63, 384.29, 143.93, 111.70, 100, 79.87, 94.74, 100.53, 434.39, 40.95, 206.66, 213.59, 139.61, 50.46, 115.56],
#    'GoalModel_MD3_BW2': [42.12, 75.87, 53.26, 47.95, 40.88, 33.42, 93.80, 51.13, 34.20, 29.15, 38.71],
#}


data_time_execution = {
    'GoalModel_MD1_BW3': [10.65, 10.78],
    'GoalModel_MD3_BW1': [10.66, 17.97, 13.60, 12.58, 11.81, 10.51, 17.45, 10.48, 11.32, 6.38, 10.23, 9.31],
    'GoalModel_MD3_BW5': [11.17, 21.30, 13.68, 13.11, 11.28, 12.52, 20.33, 9.81, 12.97, 7.82, 12.20],
    'GoalModel_MD5_BW3': [10.95, 21.50, 14.37, 12.33, 11.99, 11.31, 20.29, 18.17, 10.43, 9.61, 10.33, 12.76],
    'GoalModel_MD1_BW1': [10.43, 10.49],
    'GoalModel_MD3_BW3': [10.79, 20.75, 13.23, 11.79, 11.12, 11.20, 20.68, 18.86, 9.52, 10.37, 12.77],
    'GoalModel_MD99_BW99': [12.53, 30.37, 15.89, 14.42, 13.18, 12.37, 13.72, 13.95, 27.72, 9.59, 20.34, 30.01, 17.09, 7.98, 14.95],
    'GoalModel_MD3_BW2': [10.70, 21.69, 14.11, 11.55, 11.61, 10.37, 20.29, 14.41, 10.45, 10.85, 10.08],
}


colors = {
    1: "r",
    3: "g",
    5: "b",
    99: "k"
}

markers = {
    1: "o",
    2: "X",
    3: "s",
    5: "P",
    99: "D",
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate mean execution time for each goal model
mean_execution_times = {model: sum(times) / len(times) for model, times in data_time_execution.items()}

# Add mean execution time to the DataFrame
df["Mean Execution Time (s)"] = list(mean_execution_times.values())


# Calculate relevance as a percentage: ((Total entailing triples selected for refinement) / (Total unique entailing triples)) * 100
df['Relevance (%)'] = (df['Total entailing triples selected for refinement'] / df['Number of unique entailing triples']) * 100

print(df.to_string())

# Find the optimal parameters
optimal_config = df.sort_values(by=['Relevance (%)', 'Number of unique entailing triples', 'Mean Execution Time (s)'], ascending=[False, False, True]).iloc[0]
print("Optimal configuration based on highest relevance and coverage with lower execution time:")
print(optimal_config)

# Execution time vs relevance plot
fig1, ax1 = plt.subplots()
plt.grid()
for idx, row in df.iterrows():
    ax1.scatter(row["Mean Execution Time (s)"], row["Relevance (%)"], c=colors[row["Max depth"]], marker=markers[row["Beam width"]], s=80)

ax1.set_xlabel("Mean Execution Time (s)", fontsize='large')
ax1.set_xlim([7, 17.5])
ax1.set_ylabel("Relevance (%)", fontsize='large')
ax1.set_ylim([0, 50])

md_title = Line2D([0], [0], ls="", label='Max depth')
md_1_lgd = mpatches.Patch(color='r', label='1')
md_3_lgd = mpatches.Patch(color='g', label='3')
md_5_lgd = mpatches.Patch(color='b', label='5')
md_99_lgd = mpatches.Patch(color='k', label='99')

line_skip = Line2D([0], [0], ls="", label="", visible=False)

bw_title = Line2D([0], [0], ls="", label='Beam width')
bw_1_lgd = Line2D([0], [0], marker="o", ms=8, color='k', ls="", label='1')
bw_2_lgd = Line2D([0], [0], marker="X", ms=8, color='k', ls="", label='2')
bw_3_lgd = Line2D([0], [0], marker="s", ms=8, color='k', ls="", label='3')
bw_5_lgd = Line2D([0], [0], marker="P", ms=8, color='k', ls="", label='5')
bw_99_lgd = Line2D([0], [0], marker="D", ms=8, color='k', ls="", label='99')

ax1.legend(handles=[md_title, md_1_lgd, md_3_lgd, md_5_lgd, md_99_lgd, line_skip, bw_title, bw_1_lgd, bw_2_lgd, bw_3_lgd, bw_5_lgd, bw_99_lgd], fontsize='large')

plt.tight_layout()
#plt.savefig("./relevance_time_tradeoff.eps")
plt.savefig("./relevance_time_tradeoff_local.eps")
plt.show()

# Execution time vs NUET plot
fig2, ax2 = plt.subplots()
plt.grid()
for idx, row in df.iterrows():
    ax2.scatter(row["Mean Execution Time (s)"], row["Number of unique entailing triples"], c=colors[row["Max depth"]], marker=markers[row["Beam width"]], s=80)

ax2.set_xlabel("Mean Execution Time (s)", fontsize='large')
ax2.set_xlim([7, 17.5])
ax2.set_ylabel("Number of unique entailing triples", fontsize='large')
ax2.set_ylim([0, 80])

ax2.legend(handles=[md_title, md_1_lgd, md_3_lgd, md_5_lgd, md_99_lgd, line_skip, bw_title, bw_1_lgd, bw_2_lgd, bw_3_lgd, bw_5_lgd, bw_99_lgd], fontsize='large')

plt.tight_layout()
#plt.savefig("./nuet_time_tradeoff.eps")
plt.savefig("./nuet_time_tradeoff_local.eps")
plt.show()

# Execution time vs number of leaf goals plot
fig3, ax3 = plt.subplots()
plt.grid()
for idx, row in df.iterrows():
    ax3.scatter(row["Mean Execution Time (s)"], row["Number of leaf goals"], c=colors[row["Max depth"]], marker=markers[row["Beam width"]], s=80)

ax3.set_xlabel("Mean Execution Time (s)", fontsize='large')
ax3.set_xlim([7, 17.5])
ax3.set_ylabel("Number of leaf goals produced", fontsize='large')
ax3.set_ylim([0, 8])

ax3.legend(handles=[md_title, md_1_lgd, md_3_lgd, md_5_lgd, md_99_lgd, line_skip, bw_title, bw_1_lgd, bw_2_lgd, bw_3_lgd, bw_5_lgd, bw_99_lgd], fontsize='large')

plt.tight_layout()
plt.savefig("./nlg_time_tradeoff_local.eps")
plt.show()

# Execution time vs NUET plot
fig4, ax4 = plt.subplots()
plt.grid()
for idx, row in df.iterrows():
    ax4.scatter(row["Mean Execution Time (s)"], row["Number of direct or indirect refinements of a goal"], c=colors[row["Max depth"]], marker=markers[row["Beam width"]], s=80)

ax4.set_xlabel("Mean Execution Time (s)", fontsize='large')
ax4.set_xlim([7, 17.5])
ax4.set_ylabel("Number of goals produced", fontsize='large')
ax4.set_ylim([0, 15])

ax4.legend(handles=[md_title, md_1_lgd, md_3_lgd, md_5_lgd, md_99_lgd, line_skip, bw_title, bw_1_lgd, bw_2_lgd, bw_3_lgd, bw_5_lgd, bw_99_lgd], fontsize='large')

plt.tight_layout()
plt.savefig("./ng_time_tradeoff_local.eps")
plt.show()
