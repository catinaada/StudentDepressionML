import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn import tree


#Functie pentru afisarea confusion matrix
def conf_mtrx(y_test, y_pred,model_name): 
    cm = confusion_matrix(y_test,y_pred)    
    f, ax = plt.subplots(figsize =(5,5))
    sns.heatmap(cm,annot = True, linewidths=0.5, linecolor="red",fmt = ".0f",ax=ax)
    plt.xlabel("predicted y values")
    plt.ylabel("real y values")
    plt.title("\nConfusion Matrix "+ model_name)
    st.pyplot(f)


#Functie pentru afisarea curbei ROC si AUC
def roc_auc_curve_plot(model_name, X_testt, y_testt):
    ns_probs = [0 for _ in range(len(y_testt))]
    model_probs = model_name.predict_proba(X_testt)
    model_probs = model_probs[:, 1]
    ns_auc = roc_auc_score(y_testt, ns_probs)
    lr_auc = roc_auc_score(y_testt, model_probs)
    st.write(f"No Skill: ROC AUC = {ns_auc:.3f}")
    st.write(f"Classifier: ROC AUC = {lr_auc:.3f}")
    ns_fpr, ns_tpr, _ = roc_curve(y_testt, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(y_testt, model_probs)
    plt.figure(figsize=(7, 7))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='Clasifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    st.pyplot(plt)


#####################################################################

st.title('Analiza starii mentale a studentilor')

# Setăm stilul pentru grafic
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None

# Citim datele
df = pd.read_csv("Depression Student Dataset.csv")


st.subheader('Partea I: Analiza datelor')

# Afișăm primele 10 rânduri
st.write("Primele 10 randuri din setul de date:",df.head(10))

# Verificăm dimensiunile datasetului
st.write("Dimensiunea datasetului:", df.shape)

# Afișăm coloanele numerice
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
st.write("Coloane numerice:", numeric_cols)

# Afișăm coloanele non-numerice
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
st.write("Coloane non-numerice:", non_numeric_cols)

st.markdown("""
            ###  Curățarea datelor:
            - *eliminarea înregistrărilor lipsă;*
            - *aproximarea valorilor prin medie;*
            - *eliminarea duplicatelor.*
            """)
st.subheader('Heatmap pentru valorile lipsă')
cols = df.columns
colours = ['skyblue', '#cb4017'] # albastru pentru valoare existentă, galben pentru valoare lipsă
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
st.pyplot(plt)

fig, ax = plt.subplots(figsize=(15, 4))
df.isna().mean().sort_values().plot(
     kind="bar", ax=ax,color='skyblue',
       title="Procentajul valorilor lipsă pe caracteristică",
       ylabel="Procentajul valorilor lipsă pe caracteristică" )
st.pyplot(fig)

st.write("Înlocuim valorile lipsă cu media pentru coloanele numerice și cu valoarea cea mai frecventă pentru cele non-numerice.")
for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:
        med = df[col].mean()
        df[col] = df[col].fillna(med)

for col in non_numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:
        top = df[col].describe()['top']
        df[col] = df[col].fillna(top)

# Afișăm primele 10 rânduri după completarea valorilor lipsă
st.write("Primele 10 rânduri după completarea valorilor lipsă:",df.head(10))

# Eliminăm duplicatele
st.write(f"Numărul de duplicate eliminate: {df.duplicated().sum()}")
df.drop_duplicates(subset=None, inplace=True,ignore_index=False)

# copie date curatate
dff=df.copy()

st.markdown(""" 
           ### Aplicarea unor statistici descriptive
          -	*Agregări (summary)* 
          -	*Matricea de corelație;* 
          -	*Reprezentări grafice.* 
            """)

st.write("Descrierea statistică a setului de date:",df.describe(include="all"))
st.markdown("""
            Din aceste statistici putem observa urmatoarele aspecte:
         - Din cei 502 participanți, 53% sunt de gen masculin.
         - Vârsta medie este de aproximativ 26 de ani, cu o variație semnificativă.
         - Presiunea academică percepută este moderată, iar satisfacția cu studiile este ușor peste medie.
         - Majoritatea participanților dorm 7-8 ore pe noapte.
         - Nu avem detalii despre categoriile de obiceiuri alimentare, dar majoritatea se încadrează în categoria Moderate.
         - Aproximativ jumătate dintre participanți au avut gânduri suicidare.
         - Numărul mediu de ore de studiu este de aproximativ 6.4 pe săptămână.
         - Nivelul de stres financiar este moderat.
         - Există o istorie familială de boli mintale la o parte semnificativă dintre participanți.
         - Majoritatea participanților au raportat simptome de depresie.
        """)

#Histograma
numeric_cols_dID = df.select_dtypes(include=[np.number]).drop(columns=['ID']).columns
st.subheader('Histogramă pentru variabila numerică selectată')
selected_col_h = st.selectbox("Selectează o variabilă numerică:", numeric_cols_dID,key='hist_num_selector')
 
# Generăm histograma pentru variabila selectată 
if selected_col_h: 
    fig, ax = plt.subplots()
    df[selected_col_h].plot(kind='hist', bins=100, ax=ax,color='skyblue', edgecolor="black")
    ax.set_title(f'Histograma pentru {selected_col_h}') 
    ax.set_xlabel(selected_col_h) 
    ax.set_ylabel('Frecvență') 
    st.pyplot(fig)


st.subheader("Bar Plot pentru variabilele categoriale:")
categorical_cols = ['Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
                    'Family History of Mental Illness', 'Sleep Duration','Depression']

selected_col_hnn = st.selectbox("Selectează o variabilă categorială:", categorical_cols,key='barPlot_nn_selector')
if selected_col_hnn:
    fig, ax = plt.subplots()
    df[selected_col_hnn].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f'Bar Plot pentru {selected_col_hnn}')
    ax.set_ylabel('Frecvența')
    ax.set_xlabel(selected_col_hnn)
    st.pyplot(fig)


st.subheader("Distribuția variabilei Depression în funcție de o variabilă selectată")

# Combinăm toate coloanele fără 'Depression' si fara ID
exclude_columns = ['Depression', 'ID']  # Lista cu coloanele pe care vrei să le excluzi
all_columns = [col for col in df.columns if col not in exclude_columns]
# all_columns = [col for col in df.columns if col != 'Depression']
selected_col = st.selectbox("Selectează o variabilă:", all_columns, key='dist_by_depression')

if selected_col:
    if selected_col in numeric_cols_dID:
        fig, ax = plt.subplots()
        sns.countplot(data=df, hue='Depression', x=selected_col, ax=ax, palette="Blues")
        ax.set_title(f'Distribuția variabilei {selected_col} în funcție de Depression')
        st.pyplot(fig)
    elif selected_col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=selected_col, hue='Depression', ax=ax, palette="Blues")
        ax.set_title(f'Distribuția variabilei {selected_col} în funcție de Depression')
        st.pyplot(fig)
        

st.write("Pe măsură ce nivelul presiunii academice crește, există o tendință mai mare ca persoanele să fie afectate de depresie. Acest lucru arată o posibilă relație între stresul academic și sănătatea mentală.")
st.write("In plus, graficul evidențiază o corelație puternică între depresie și gândurile suicidale. Persoanele care suferă de depresie sunt mult mai susceptibile să fi avut gânduri suicidale în comparație cu cele care nu suferă de depresie.")
st.write("Persoanele care au raportat niveluri mai scăzute de satisfacție față de studiu au o frecvență mai mare a depresiei comparativ cu cele care au raportat niveluri mai ridicate de satisfacție, ceea ce sugerează o posibilă legătură negativă între depresie și satisfacția în studii.")     
st.write("Mai mult, graficul sugerează o posibilă asociere între nivelul de stres financiar și prezența depresiei. Persoanele care raportează niveluri mai mari de stres financiar par să aibă o probabilitate ușor mai mare de a raporta și simptome de depresie.")
st.write("Depresia este mai frecventă la persoanele mai tinere, în special în jurul vârstei de 18-22 de ani, în timp ce incidența depresiei scade la persoanele mai în vârstă (peste 30 de ani), acest lucru sugerand o posibilă asociere între vârstă și depresie, tinerii fiind mai vulnerabili.")
st.write("")


st.markdown("""
           ###  Îmbunătățirea acurateței datelor:
          - *conversii între diferite tipuri*
          - *detectarea valorilor anormale (outliers)*
          - *prelucrarea șirurilor de caractere*
            """)

# Afișăm boxplot pentru fiecare variabila numerica
st.subheader('Boxplot pentru variabila numerica selectata')
selected_col_b = st.selectbox("Selectează o variabilă numerică:", numeric_cols_dID,key='boxplot_selector')
if selected_col_b:
    fig, ax = plt.subplots()
    df.boxplot(column=[selected_col_b], ax=ax,color='#5353EC')
    ax.set_title(f'Boxplot pentru {selected_col_b}')
    st.pyplot(fig)

st.write("Interpretare: Nu exista outliers.")


st.subheader("Prelucrarea șirurilor de caractere")

# Afiseaza denumirile coloanelor
st.write("Tipul de date al coloanelor din setul de date:") 
st.write(df.dtypes)

labelEncoder = LabelEncoder()
df['Gender'] = labelEncoder.fit_transform(df['Gender'].astype(str))
st.write("Valori unice în coloana 'Gender':", df['Gender'].unique())

df['Dietary Habits'] = labelEncoder.fit_transform(df['Dietary Habits'].astype(str))
st.write("Valori unice în coloana 'Dietary Habits':", df['Dietary Habits'].unique())

df['Family History of Mental Illness'] = labelEncoder.fit_transform(df['Family History of Mental Illness'].astype(str))
st.write("Valori unice în coloana 'Family History of Mental Illness':", df['Family History of Mental Illness'].unique())

df['Have you ever had suicidal thoughts ?'] = labelEncoder.fit_transform(df['Have you ever had suicidal thoughts ?'].astype(str))
st.write("Valori unice în coloana 'Have you ever had suicidal thoughts ?':", df['Have you ever had suicidal thoughts ?'].unique())

df['Depression'] = labelEncoder.fit_transform(df['Depression'].astype(str))
st.write("Valori unice în coloana 'Depression':", df['Depression'].unique())

df['Age'] = df['Age'].astype(int)
df['Study Hours'] = df['Study Hours'].astype(int)
df['Financial Stress'] = df['Financial Stress'].astype(int)
df['Academic Pressure'] = df['Academic Pressure'].astype(int)

# Aplică maparea pentru Sleep Duration
mapping = { 'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3 }
df['Sleep Duration Discrete'] = df['Sleep Duration'].map(mapping)

# Crează o copie a DataFrame-ului
dfc = df.copy()

# Șterge coloana și actualizează DataFrame-ul
dfc.drop('Sleep Duration', axis=1, inplace=True)

# Afișează setul de date modificat
st.write("Primele 5 randuri dupa modificare:",dfc.head(10))

#corelatia dintre depresie si restul variabilelor
corr = dfc.corr(method='pearson')
corr.sort_values(["Depression"], ascending = False, inplace = True)
st.write("Corelatia dintre depresie si celelalte variabile:",corr.Depression)


st.write("Heatmap al corelațiilor")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(dfc.corr(), annot=True, cmap='coolwarm', ax=ax) 
ax.set_title('Heatmap al corelațiilor') 
st.pyplot(fig)

#########################################################
### TRANSFORMAREA DATELOR PENTRU APLICAREA MODELELOR
X = dfc.drop('Depression', axis=1)
y = dfc['Depression']

## Selectarea atributelor importante - Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression
k_features = 6
selector = SelectKBest(f_regression, k = k_features)
X_new = selector.fit(X, y)
names = X.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
st.write("Selectarea atributelor importante:",ns_df_sorted)
cols=names
X = selector.fit_transform(X, y)
X = pd.DataFrame(X, columns=cols)


cols=X.columns.tolist()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(data=X, columns=cols)

# Construirea seturilor de date de train si test: 80% train si 20% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.20)

st.write("In continuare, se vor aplica modelele de clasificare pentru a previziona aparitia depresiei in functie de cele mai importante 6 atribute.")

### APLICAREA MODELELOR DE CLASIFICARE

st.subheader("Partea II: Aplicarea modelelor de clasificare")

# Retinem acuratetea modelelor intr-un dataframe
acc_df = pd.DataFrame(columns = ['Model','Accuracy', 'F1-Score'])

#### Aplicarea modelului de REGRESIE LOGISTICA
st.subheader('1. Aplicarea modelului de Regresie Logistica')
RL = LogisticRegression(max_iter=100, multi_class='auto', penalty='l2',  solver='lbfgs')
RL.fit(X_train, y_train)
y_predicted = RL.predict(X_test)

# Verificarea modelului de regresie logistica. 
# Afisam acuratetea, matricea de confuzie şi raportul de clasificare

cm=confusion_matrix(y_test, y_predicted)
st.write("Confusion matrix:")
st.write(cm)
conf_mtrx(y_test, y_predicted, 'RL')

RL_report=classification_report(y_test, y_predicted, output_dict=True)
st.write("### Classification Report:")
st.dataframe(pd.DataFrame(RL_report).transpose())

acuratetea_RL=accuracy_score(y_test, y_predicted)
st.write("### Acuratetea RL: ")
st.write(acuratetea_RL) 

st.write("Modelul a clasificat corect majoritatea observatiilor, avand acuratete foarte mare. ")

# Graficul ROC AUC
st.write("### ROC AUC Curve:")
roc_auc_curve_plot(RL, X_test, y_test)  

# Adăugarea metricei în DataFrame-ul acc_df
if 'acc_df' not in st.session_state:
    st.session_state.acc_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score'])

st.session_state.acc_df.loc[len(st.session_state.acc_df)] = [
    'RL', 
    accuracy_score(y_test, y_predicted), 
    f1_score(y_test, y_predicted, average='weighted')
]

st.write("Conform ROC AUC, modelul este extrem de performant.")

# Afișarea DataFrame-ului cu metricile
st.write("### Rezumatul performantei modelului:")
st.dataframe(st.session_state.acc_df)

st.write("Intrucat valorile acuratetei si ale F1-score sunt mari, modelul este unul foarte performant.")

# Predictie
st.write("### Predictie : Aplicarea modelului RL din setul de test:")
single_predicted = RL.predict(X_test[2:3])
st.write('Actual: ',int(y_test[2:3]))
st.write('Estimat: ' ,int(single_predicted))
st.write('Probabilitate 0/1:', RL.predict_proba(X_test[2:3]) )

st.write("Modelul de regresie logistica a facut o predictie corecta si cu o incredere foarte mare.")

#### Aplicarea modelului Decission Tree Classifier
st.write("### 2. Aplicarea modelului Decission Tree Classifier: ")

dtc = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_predicted = dtc.predict(X_test)

cm=confusion_matrix(y_test, y_predicted)
st.write("Confusion matrix:")
st.write(cm)
conf_mtrx(y_test, y_predicted, 'DTC')

dtc_report=classification_report(y_test, y_predicted, output_dict=True)
st.write("### Classification Report:")
st.dataframe(pd.DataFrame(dtc_report).transpose())

acuratetea_DTC=accuracy_score(y_test, y_predicted)
st.write("### Acuratetea DTC: ")
st.write(acuratetea_DTC)  # Formatăm pentru a afișa doar 4 zecimale

st.write("Modelul a clasificat corect majoritatea observatiilor, avand acuratete mare.")

# Graficul ROC AUC
st.write("### ROC AUC Curve:")
roc_auc_curve_plot(dtc, X_test, y_test)  # Funcția trebuie să fie definită deja, cu `st.pyplot` pentru afișare

# Adăugarea metricei în DataFrame-ul acc_df
if 'acc_df' not in st.session_state:
    st.session_state.acc_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score'])

st.session_state.acc_df.loc[len(st.session_state.acc_df)] = [
    'DTC', 
    accuracy_score(y_test, y_predicted), 
    f1_score(y_test, y_predicted, average='weighted')
]

st.write("Conform ROC AUC, modelul este performant.") 

# Afișarea DataFrame-ului cu metricile
st.write("### Rezumatul performantei modelului:")
st.dataframe(st.session_state.acc_df)

st.write("Intrucat valorile acuratetei si ale F1-score sunt mari, modelul este unul performant.")

# Predictie
st.write("### Predictie : Aplicarea modelului DTC din setul de test:")
single_predicted = dtc.predict(X_test[2:3])
st.write('Actual: ',int(y_test[2:3]))
st.write('Estimat: ' ,int(single_predicted))
st.write('Probabilitate 0/1:', dtc.predict_proba(X_test[2:3]) )

st.write("Modelul DTC a facut o predictie perfecta, corecta si cu o incredere foarte mare.")

###  Retele neuronale MLPClassifier
st.header("3. Retele neuronale MLPClassifier")

mlpc = MLPClassifier(solver='adam', alpha=0.0005,hidden_layer_sizes=(25,), activation ='logistic', max_iter=1000)
mlpc.fit(X_train, y_train)
y_predicted = mlpc.predict(X_test)

cm=confusion_matrix(y_test, y_predicted)
st.write("Confusion matrix:")
st.write(cm)
conf_mtrx(y_test, y_predicted, 'MLPC')


mlpc_report=classification_report(y_test, y_predicted, output_dict=True)
st.write("### Classification Report:")
st.dataframe(pd.DataFrame(mlpc_report).transpose())

acuratetea_mlpc=accuracy_score(y_test, y_predicted)
st.write("### Acuratetea MLPC: ")
st.write(acuratetea_mlpc)  # Formatăm pentru a afișa doar 4 zecimale

st.write("Modelul a clasificat corect majoritatea observatiilor, avand acuratete foarte mare.")

# Graficul ROC AUC
st.write("### ROC AUC Curve:")
roc_auc_curve_plot(mlpc, X_test, y_test)  # Funcția trebuie să fie definită deja, cu `st.pyplot` pentru afișare

# Adăugarea metricei în DataFrame-ul acc_df
if 'acc_df' not in st.session_state:
    st.session_state.acc_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score'])

st.session_state.acc_df.loc[len(st.session_state.acc_df)] = [
    'MLPC', 
    accuracy_score(y_test, y_predicted), 
    f1_score(y_test, y_predicted, average='weighted')
]

st.write("Conform ROC AUC, modelul este extrem de performant.")

# Predictie
st.write("### Predictie : Aplicarea modelului MLPC din setul de test:")
single_predicted = mlpc.predict(X_test[2:3])
st.write('Actual: ',int(y_test[2:3]))
st.write('Estimat: ' ,int(single_predicted))
st.write('Probabilitate 0/1:', mlpc.predict_proba(X_test[2:3]) )

st.write("Modelul MLPC a facut o predictie corecta si cu o incredere foarte mare.")

# Afișarea DataFrame-ului cu metricile
st.write("### Rezumatul performantei modelului:")
st.dataframe(st.session_state.acc_df)

st.header("CONCLUZIE")
st.subheader("Conform rezultatelor obtinute de fiecare model de clasificare, cel de Regresie Logistica s-a dovedit a fi cel mai bun, fiind urmat de modelele MLPC si DTC.")
st.subheader("Asadar, tinerii cu un nivel crescut al presiunii academice, cu ganduri suicidale, un nivel scazut de satisfactie in studiat si care sunt stresati din punct de vedere financiar sunt mai predispusi sa sufere de depresie.")