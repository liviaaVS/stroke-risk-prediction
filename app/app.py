import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Garante que o Python encontre o módulo 'pipeline' no momento do joblib.load
sys.path.append(os.path.abspath("src"))
import pipeline

# -----------------------------------------------------------------------------
# Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Predição de AVC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Estilização CSS Personalizada (Adaptativa para Tema Light e Dark - Paleta Sóbria)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Geral - Herda cor nativa do tema (Light ou Dark) */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Cartões e painéis com fundos neutros adaptativos que funcionam perfeitamente em Light e Dark Mode */
    .solid-card {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .header-panel {
        background-color: rgba(128, 128, 128, 0.09);
        border: 1px solid rgba(128, 128, 128, 0.24);
        padding: 24px 32px;
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .header-panel h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .header-panel p {
        margin: 8px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.8;
        line-height: 1.5;
    }

    /* Indicadores */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-low { background-color: rgba(77, 138, 84, 0.15); color: #2e6b34; border: 1px solid rgba(77, 138, 84, 0.4); }
    .badge-high { background-color: rgba(207, 107, 107, 0.15); color: #b83232; border: 1px solid rgba(207, 107, 107, 0.4); }
    
    @media (prefers-color-scheme: dark) {
        .badge-low { color: #68b072; }
        .badge-high { color: #cf6b6b; }
    }

    /* Botões sóbrios na cor terracota do Claude */
    div[data-testid="stButton"] button {
        background-color: #d97757;
        color: #ffffff !important;
        border: 1px solid #c2694b;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #c86b4b;
        border-color: #b35d40;
        color: #ffffff !important;
    }

    /* Abas nativas limpas com transparência adaptativa */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 4px;
        border-radius: 6px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        opacity: 0.8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(128, 128, 128, 0.16) !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carregamento de Recursos e Modelos
# -----------------------------------------------------------------------------
@st.cache_resource
def load_stroke_model():
    model_path = os.path.join("models", "stroke_risk_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_preprocessor():
    prep_path = os.path.join("data", "processed", "preprocessor.pkl")
    if not os.path.exists(prep_path):
        return None
    return joblib.load(prep_path)

@st.cache_data
def load_metrics_and_roc():
    json_path = os.path.join("reports", "metrics.json")
    roc_path = os.path.join("reports", "roc_curve_test.csv")
    raw_path = os.path.join("data", "raw", "healthcare-dataset-stroke-data.csv")
    
    metrics = None
    roc_df = None
    raw_df = None
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        
    return metrics, roc_df, raw_df

artifact = load_stroke_model()
preprocessor = load_preprocessor()
metrics_json, roc_df, raw_df = load_metrics_and_roc()

# -----------------------------------------------------------------------------
# Navegação Principal (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Predição de Risco de AVC")
    st.caption("Sistema quantitativo de suporte à decisão")
    st.markdown("---")
    
    page = st.radio(
        "Módulo de Navegação:",
        ["Calculadora de Risco", "Desempenho do Modelo", "Metodologia e Pipeline"],
        index=0
    )
    
    st.markdown("---")
    if artifact is not None:
        threshold_val = artifact.get('threshold', 0.7278)
        st.markdown(f"**Status:** Modelo ativo<br>**Limiar de corte:** `{threshold_val:.4f}`", unsafe_allow_html=True)
    else:
        st.error("Modelo não encontrado em models/stroke_risk_model.joblib.")
        
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("Métodos Quantitativos | TADS 2026.1")

# =============================================================================
# MÓDULO 1: CALCULADORA DE RISCO
# =============================================================================
if page == "Calculadora de Risco":
    st.markdown("""
    <div class="header-panel">
        <h1>Calculadora de Risco Clínico</h1>
        <p>Informe os dados biométricos e variáveis clínicas para calcular a probabilidade estimada de Acidente Vascular Cerebral (AVC) utilizando o modelo de Regressão Logística com balanceamento SMOTE.</p>
    </div>
    """, unsafe_allow_html=True)

    if artifact is None:
        st.warning("O modelo ainda não foi gerado. Execute o script src/train.py para treinar o modelo.")
    else:
        model = artifact['model']
        threshold = artifact['threshold']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""<div class="solid-card"><h4>Dados Demográficos e Sociais</h4>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Idade (anos)", min_value=0, max_value=120, value=55, step=1)
                gender = st.selectbox("Gênero", ["Male", "Female", "Other"], index=0, format_func=lambda x: "Masculino" if x=="Male" else ("Feminino" if x=="Female" else "Outro"))
            with c2:
                ever_married = st.selectbox("Histórico matrimonial", ["Yes", "No"], index=0, format_func=lambda x: "Casado(a) ou ex-casado(a)" if x=="Yes" else "Nunca casado(a)")
                residence_type = st.selectbox("Tipo de residência", ["Urban", "Rural"], index=0, format_func=lambda x: "Urbana" if x=="Urban" else "Rural")
                
            work_type = st.selectbox(
                "Categoria ocupacional", 
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                format_func=lambda x: {
                    "Private": "Setor privado", "Self-employed": "Trabalho autônomo", 
                    "Govt_job": "Serviço público", "children": "Estudante ou criança", 
                    "Never_worked": "Sem histórico laboral"
                }[x]
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class="solid-card"><h4>Parâmetros Biométricos e Clínicos</h4>""", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                avg_glucose = st.number_input("Glicemia média (mg/dL)", min_value=30.0, max_value=400.0, value=115.5, step=1.0)
                hypertension = st.selectbox("Hipertensão arterial", [0, 1], format_func=lambda x: "Sim (Diagnóstico formal)" if x==1 else "Não ausente")
            with c4:
                bmi = st.number_input("Índice de Massa Corporal (IMC)", min_value=10.0, max_value=70.0, value=28.4, step=0.5)
                heart_disease = st.selectbox("Histórico de doença cardíaca", [0, 1], format_func=lambda x: "Sim (Diagnóstico prévio)" if x==1 else "Não ausente")
                
            smoking_status = st.selectbox(
                "Status de tabagismo",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                format_func=lambda x: {
                    "never smoked": "Nunca fumou", "formerly smoked": "Ex-fumante",
                    "smokes": "Fumante ativo", "Unknown": "Informação indisponível"
                }[x]
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            calculate_btn = st.button("Calcular probabilidade", use_container_width=True)

        if calculate_btn:
            input_df = pd.DataFrame([{
                "id": 99999,
                "gender": gender,
                "age": float(age),
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": float(avg_glucose),
                "bmi": float(bmi),
                "smoking_status": smoking_status
            }])
            
            try:
                # Se o modelo não possui pipeline embutido, transforma os dados com o preprocessor carregado
                if preprocessor is not None and not hasattr(model, "named_steps") and not hasattr(model, "steps"):
                    transformed_arr = preprocessor.transform(input_df)
                    feature_names = artifact.get('features') if (artifact and isinstance(artifact, dict) and 'features' in artifact) else (
                        preprocessor.named_steps['column_transformer'].get_feature_names_out() if hasattr(preprocessor, 'named_steps') and 'column_transformer' in preprocessor.named_steps else None
                    )
                    if feature_names is not None:
                        input_df = pd.DataFrame(transformed_arr, columns=feature_names)
                    else:
                        input_df = transformed_arr

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    stroke_prob = probs[1]
                else:
                    decision = model.decision_function(input_df)[0]
                    stroke_prob = 1 / (1 + np.exp(-decision))
                
                st.markdown("---")
                res_col1, res_col2 = st.columns([1.2, 1.8])
                
                with res_col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = stroke_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "<b>Probabilidade de AVC</b>", 'font': {'size': 16}},
                        number = {'suffix': "%", 'font': {'size': 38}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                            'bar': {'color': "#d97757"},
                            'bgcolor': "rgba(128, 128, 128, 0.12)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(128, 128, 128, 0.3)",
                            'steps': [
                                {'range': [0, threshold * 100], 'color': "rgba(128, 128, 128, 0.06)"},
                                {'range': [threshold * 100, 100], 'color': "rgba(217, 119, 87, 0.2)"}
                            ],
                            'threshold': {
                                'line': {'color': "#d97757", 'width': 3},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'family': "Inter"},
                        height=280,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with res_col2:
                    st.markdown("""<div class="solid-card" style="min-height: 280px; display: flex; flex-direction: column; justify-content: center;">""", unsafe_allow_html=True)
                    
                    if stroke_prob >= threshold:
                        st.markdown("""
                        <span class="badge badge-high">Classificação: Risco Elevado</span>
                        <h3 style="color: #b83232; margin-top: 12px; margin-bottom: 8px;">Indicador de probabilidade superior ao limiar</h3>
                        <p style="opacity: 0.8; font-size: 0.95rem;">A probabilidade calculada (<b>{:.1f}%</b>) excede o limiar de corte ótimo ajustado na validação (<b>{:.1f}%</b>).</p>
                        """.format(stroke_prob*100, threshold*100), unsafe_allow_html=True)
                        
                        st.markdown("""
                        **Condutas sugeridas:**
                        - Avaliação cardiológica e neurológica clínica detalhada.
                        - Acompanhamento laboratorial de glicemia e aferição contínua da pressão arterial.
                        - Revisão de fatores de risco modificáveis e estilo de vida.
                        """)
                    else:
                        st.markdown("""
                        <span class="badge badge-low">Classificação: Risco Controlado</span>
                        <h3 style="color: #2e6b34; margin-top: 12px; margin-bottom: 8px;">Indicador dentro da margem esperada</h3>
                        <p style="opacity: 0.8; font-size: 0.95rem;">A probabilidade calculada (<b>{:.1f}%</b>) encontra-se abaixo do limiar de corte ajustado na validação (<b>{:.1f}%</b>).</p>
                        """.format(stroke_prob*100, threshold*100), unsafe_allow_html=True)
                        
                        st.markdown("""
                        **Condutas sugeridas:**
                        - Manutenção de hábitos saudáveis, atividade física regular e controle de peso.
                        - Monitoramento periódico de rotina de acordo com a faixa etária.
                        """)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Erro no processamento da inferência: {str(e)}")

# =============================================================================
# MÓDULO 2: DESEMPENHO DO MODELO
# =============================================================================
elif page == "Desempenho do Modelo":
    st.markdown("""
    <div class="header-panel">
        <h1>Análise de Desempenho e Métricas</h1>
        <p>Apresentação quantitativa dos resultados do modelo no conjunto de teste independente (1.022 amostras), incluindo curva ROC e matriz de confusão.</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics_json is not None:
        test_m = metrics_json.get('test', {})
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        
        with t_col1:
            st.metric("ROC AUC", f"{test_m.get('roc_auc', 0.8409):.4f}")
        with t_col2:
            st.metric("Recall (Sensibilidade)", f"{test_m.get('recall', 0.6000):.2%}")
        with t_col3:
            st.metric("Precisão", f"{test_m.get('precision', 0.2055):.2%}")
        with t_col4:
            st.metric("F1-Score", f"{test_m.get('f1_score', 0.3061):.4f}")
        with t_col5:
            st.metric("Limiar de Corte", f"{test_m.get('threshold', 0.7278):.4f}")
            
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("""<div class="solid-card"><h4>Curva ROC - Conjunto de Teste</h4>""", unsafe_allow_html=True)
        if roc_df is not None:
            fig_roc = px.line(
                roc_df, x='false_positive_rate', y='true_positive_rate',
                labels={'false_positive_rate': 'Taxa de Falsos Positivos (1 - Especificidade)', 'true_positive_rate': 'Taxa de Verdadeiros Positivos (Recall)'}
            )
            fig_roc.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=1, y1=1)
            fig_roc.update_traces(line_color='#d97757', line_width=2.5)
            fig_roc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(128, 128, 128, 0.04)",
                font={'family': "Inter"}, height=350,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("Arquivo roc_curve_test.csv não encontrado.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("""<div class="solid-card"><h4>Matriz de Confusão - Conjunto de Teste</h4>""", unsafe_allow_html=True)
        if metrics_json is not None:
            cm = metrics_json.get('test', {}).get('confusion_matrix', {'tn': 839, 'fp': 133, 'fn': 16, 'tp': 34})
            cm_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Classificação do Modelo", y="Observação Real", color="Contagem"),
                x=['Não AVC (0)', 'AVC (1)'],
                y=['Não AVC (0)', 'AVC (1)'],
                text_auto=True,
                color_continuous_scale='Oranges'
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={'family': "Inter"}, height=350,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if raw_df is not None:
        st.markdown("### Análise Exploratória e Distribuição de Riscos")
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("""<div class="solid-card"><h4>Distribuição Etária por Ocorrência de AVC</h4>""", unsafe_allow_html=True)
            fig_age = px.histogram(
                raw_df, x="age", color="stroke", nbins=30, barmode="overlay",
                labels={"age": "Idade (anos)", "stroke": "Ocorrência"},
                color_discrete_map={0: "rgba(128, 128, 128, 0.4)", 1: "#d97757"}
            )
            fig_age.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(128, 128, 128, 0.04)",
                font={'family': "Inter"}, height=320,
                legend_title="AVC (0=Não, 1=Sim)",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with eda_col2:
            st.markdown("""<div class="solid-card"><h4>Nível de Glicemia por Ocorrência de AVC</h4>""", unsafe_allow_html=True)
            fig_glu = px.box(
                raw_df, x="stroke", y="avg_glucose_level", color="stroke",
                labels={"stroke": "Ocorrência", "avg_glucose_level": "Glicemia média (mg/dL)"},
                color_discrete_map={0: "rgba(128, 128, 128, 0.4)", 1: "#d97757"}
            )
            fig_glu.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(128, 128, 128, 0.04)",
                font={'family': "Inter"}, height=320, showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_glu, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MÓDULO 3: METODOLOGIA E PIPELINE
# =============================================================================
elif page == "Metodologia e Pipeline":
    st.markdown("""
    <div class="header-panel">
        <h1>Metodologia e Arquitetura do Projeto</h1>
        <p>Descrição técnica detalhada do fluxo de processamento e das 5 etapas fundamentais do ciclo de vida dos dados.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="solid-card">
        <h4>Abordagem Geral do Problema</h4>
        <p style="opacity: 0.8; line-height: 1.6;">
        A modelagem clínica de predição de risco de Acidente Vascular Cerebral apresenta desafios estatísticos significativos, com destaque para a forte disparidade entre as classes na amostra (aproximadamente 4.8% de prevalência de AVC na base). Para evitar que o modelo atinja falsa acurácia elevada ao prever sistematicamente a classe majoritária, o pipeline foi estruturado combinando engenharia de features de discretização médica com balanceamento amostral e pesos de custo assimétrico.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("""
        <div class="solid-card">
            <h4>Fase 1: Análise Exploratória de Dados</h4>
            <ul style="opacity: 0.8; line-height: 1.6;">
                <li>Inspeção inicial de 5.110 registros e verificação das distribuições univariadas e bivariadas.</li>
                <li>Identificação de 201 valores ausentes na coluna <code>bmi</code> (Índice de Massa Corporal).</li>
                <li>Constatação da categoria não informativa <code>Unknown</code> na variável de tabagismo.</li>
                <li>Mapeamento da forte correlação não-linear entre idade avançada, hiperglicemia e ocorrência do evento de AVC.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="solid-card">
            <h4>Fase 2: Engenharia de Dados e Pré-processamento</h4>
            <ul style="opacity: 0.8; line-height: 1.6;">
                <li><b>Tratamento de nulos:</b> Imputação pela mediana em <code>bmi</code> através do <code>SimpleImputer</code>.</li>
                <li><b>Discretização (Binning):</b> Criação das variáveis categóricas <code>age_group</code> (0-18, 19-35, 36-60, 60+) e <code>glucose_group</code> (&lt;100, 100-125, &ge;126) utilizando <code>pd.cut</code> em um <code>FunctionTransformer</code> integrado.</li>
                <li><b>Codificação e Escalonamento:</b> Aplicação unificada de <code>StandardScaler</code> e <code>OneHotEncoder</code> pelo <code>ColumnTransformer</code>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div class="solid-card">
            <h4>Fase 3: Treinamento, Calibração e Avaliação</h4>
            <ul style="opacity: 0.8; line-height: 1.6;">
                <li><b>Modelagem quantitativa:</b> Regressão Logística com regularização e penalização proporcional de classes (<code>class_weight="balanced"</code>).</li>
                <li><b>Sobreamostragem sem vazamento:</b> Separação de 20% para teste independente e 20% para validação <i>antes</i> de qualquer balanceamento. O <code>SMOTE</code> é aplicado somente sobre a porção efetiva de treino (60%), garantindo que a validação e o teste preservem a prevalência real de AVC (~4.8%) usada para calibrar o limiar de decisão.</li>
                <li><b>Calibração de limiar:</b> Na partição de validação (com distribuição real), a curva Precisão-Recall é calculada para definir o limiar ótimo que maximiza o <code>F1-Score</code>.</li>
                <li><b>Métricas finais no teste:</b> Atingimento de ROC AUC de <b>0.8409</b> e Recall (sensibilidade clínica) de <b>60.00%</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="solid-card">
            <h4>Fase 4 e 5: Visualização e Aplicação Interativa</h4>
            <ul style="opacity: 0.8; line-height: 1.6;">
                <li><b>Fase 4 (BI/Analytics):</b> Gráficos exploratórios e analíticos implementados em <code>Plotly</code> para visualização interativa das curvas de decisão e distribuições de risco.</li>
                <li><b>Fase 5 (Produto Final):</b> Encapsulamento de todas as transformações em um aplicativo web <code>Streamlit</code> focado em usabilidade e sobriedade técnica, permitindo ao usuário final ou profissional de saúde executar simulações clínicas em tempo real.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
