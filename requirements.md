# ============================================
# Requirements para Projeto de ML
# Introdução a Machine Learning - 2025.2
# Previsão de Desempenho Acadêmico
# ============================================

# Manipulação de Dados
pandas==1.5.3
numpy==1.24.3

# Visualização
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Machine Learning - Scikit-learn
scikit-learn==1.2.2

# Gradient Boosting
xgboost==1.7.5
lightgbm==3.3.5

# Jupyter
jupyter==1.0.0
jupyterlab==3.6.3
ipykernel==6.22.0

# Utilidades
joblib==1.2.0
scipy==1.10.1

# Progress bars
tqdm==4.65.0

# Interpretabilidade (SHAP - para análise avançada)
shap==0.41.0

# Deploy (opcional - para bônus)
streamlit==1.22.0
flask==2.3.2

# Qualidade de Código (recomendado)
flake8==6.0.0
black==23.3.0
autopep8==2.0.2

# ============================================
# INSTRUÇÕES DE INSTALAÇÃO
# ============================================
#
# 1. Criar ambiente virtual:
#    python -m venv venv
#
# 2. Ativar ambiente:
#    - Linux/Mac: source venv/bin/activate
#    - Windows: venv\Scripts\activate
#
# 3. Instalar dependências:
#    pip install -r requirements.txt
#
# 4. Verificar instalação:
#    pip list
#
# ============================================
# VERIFICAÇÃO DE DEPENDÊNCIAS
# ============================================
#
# Para verificar se tudo foi instalado corretamente:
#
# python -c "import pandas; import numpy; import sklearn; \
#            import xgboost; import matplotlib; print('✅ All dependencies installed!')"
#
# ============================================
# NOTAS
# ============================================
#
# - Versões fixas para garantir reprodutibilidade
# - Compatível com Python 3.8+
# - Testado em Linux, macOS e Windows
# - Todas as dependências utilizadas no projeto
#
# Para atualizar para versões mais recentes:
#    pip install --upgrade <package>
#    pip freeze > requirements.txt
#
# Para remover ambiente virtual:
#    - Linux/Mac: rm -rf venv
#    - Windows: rmdir /s venv
#
# ============================================