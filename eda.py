import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(figure, dir_path, file_name, dpi=300):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    full_path = os.path.join(dir_path, file_name)
    figure.savefig(full_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)

if not os.path.exists("eda_plots"):
    os.makedirs("eda_plots")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

features = ['mean_intensity', 'std_intensity', 'skewness', 'max_intensity', 'contrast', 'energy', 'homogeneity', 'dissimilarity', 'correlation', 'entropy']
target_column = 'tumor_type'
plots_dir = "eda_plots"

print("Statistici descriptive pentru caracteristici (Date de Antrenament):")
print(train_df[features].describe())

print("Statistici descriptive pentru caracteristici (Date de Test):")
print(test_df[features].describe())

print(f"Statistici descriptive pentru target '{target_column}' (Date de Antrenament):")
print(train_df[target_column].describe())

print(f"Statistici descriptive pentru target '{target_column}' (Date de Test):")
print(test_df[target_column].describe())

# histograme pentru caracteristici
fig_train, axes_train = plt.subplots(2, 5, figsize=(25, 10)) 
fig_train.suptitle('Distributia Caracteristicilor (Date de Antrenament)', fontsize=16)
axes_train = axes_train.flatten()
for i, feature in enumerate(features):
    sns.histplot(train_df[feature], kde=True, ax=axes_train[i])
    axes_train[i].set_title(f'Distributia {feature}', fontsize=12)
    axes_train[i].set_xlabel(feature, fontsize=10)
    axes_train[i].set_ylabel('Frecventa', fontsize=10)
save_plot(fig_train, plots_dir, 'histograms_combined_train.png')

fig_test, axes_test = plt.subplots(2, 5, figsize=(25, 10))
fig_test.suptitle('Distributia Caracteristicilor (Date de Test)', fontsize=16)
axes_test = axes_test.flatten()
for i, feature in enumerate(features):
    sns.histplot(test_df[feature], kde=True, ax=axes_test[i])
    axes_test[i].set_title(f'Distributia {feature}', fontsize=12)
    axes_test[i].set_xlabel(feature, fontsize=10)
    axes_test[i].set_ylabel('Frecventa', fontsize=10)
save_plot(fig_test, plots_dir, 'histograms_combined_test.png')

# Countplots pentru target
fig_countplot_train, ax_countplot_train = plt.subplots(figsize=(8, 6))
sns.countplot(data=train_df, x=target_column, ax=ax_countplot_train, order=train_df[target_column].value_counts().index, hue=target_column, palette="coolwarm", legend=False)
ax_countplot_train.set_title(f'Distributia Target-ului: {target_column} (Antrenament)', fontsize=15)
ax_countplot_train.set_xlabel(target_column, fontsize=12)
ax_countplot_train.set_ylabel('Numar', fontsize=12)
save_plot(fig_countplot_train, plots_dir, f'countplot_{target_column}_train.png')

fig_countplot_test, ax_countplot_test = plt.subplots(figsize=(8, 6))
sns.countplot(data=test_df, x=target_column, ax=ax_countplot_test, order=test_df[target_column].value_counts().index, hue=target_column, palette="coolwarm", legend=False)
ax_countplot_test.set_title(f'Distributia Target-ului: {target_column} (Test)', fontsize=15)
ax_countplot_test.set_xlabel(target_column, fontsize=12)
ax_countplot_test.set_ylabel('Numar', fontsize=12)
save_plot(fig_countplot_test, plots_dir, f'countplot_{target_column}_test.png')

fig_boxplot_train, axes_boxplot_train = plt.subplots(2, 5, figsize=(25, 10))
fig_boxplot_train.suptitle('Boxplot-uri Caracteristici (Date de Antrenament)', fontsize=16)
axes_boxplot_train = axes_boxplot_train.flatten()
pastel_colors = sns.color_palette("pastel")
for i, feature in enumerate(features):
    sns.boxplot(data=train_df, y=feature, ax=axes_boxplot_train[i], color=pastel_colors[i % len(pastel_colors)])
    axes_boxplot_train[i].set_title(f'Boxplot {feature}', fontsize=12)
    axes_boxplot_train[i].set_ylabel(feature, fontsize=10)
save_plot(fig_boxplot_train, plots_dir, 'boxplots_combined_train.png')

correlation_matrix_train = train_df[features].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix_train, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
ax.set_title('Matrice de Corelatie a Caracteristicilor (Antrenament)', fontsize=15)
save_plot(fig, plots_dir, 'correlation_heatmap_train.png')

fig_violin_train, axes_violin_train = plt.subplots(2, 5, figsize=(25, 10))
fig_violin_train.suptitle(f'Caracteristici vs. {target_column} (Date de Antrenament)', fontsize=16)
axes_violin_train = axes_violin_train.flatten()
for i, feature in enumerate(features):
    sns.violinplot(x=target_column, y=feature, data=train_df, ax=axes_violin_train[i], hue=target_column, palette="coolwarm", legend=False)
    axes_violin_train[i].set_title(f'{feature} vs. {target_column}', fontsize=12)
    axes_violin_train[i].set_xlabel(target_column, fontsize=10)
    axes_violin_train[i].set_ylabel(feature, fontsize=10)
save_plot(fig_violin_train, plots_dir, f'violinplots_combined_train_vs_{target_column}.png')

print("EDA finalizat.")
print(f"Graficele au fost salvate in directorul '{plots_dir}'")
