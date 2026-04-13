from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

modelo = joblib.load("modelo_credito_lendingclub.pkl")
colunas = joblib.load("colunas_credito.pkl")

app = FastAPI(
    title="API — Análise de Crédito LendingClub",
    description="Modelo Random Forest treinado com dados de empréstimos 2007-2010",
    version="1.0.0"
)

class Emprestimo(BaseModel):
    politica_credito: int
    taxa_juros: float
    parcela_mensal: float
    log_renda_anual: float
    relacao_divida_renda: float
    pontuacao_fico: int
    dias_com_linha_credito: float
    saldo_rotativo: int
    utilizacao_rotativa: float
    consultas_ultimos_6meses: int
    atrasos_ultimos_2anos: int
    registros_publicos: int
    finalidade: str

@app.get("/")
def root():
    return {"status": "online", "modelo": "Random Forest", "versao": "1.0.0"}

@app.get("/app", response_class=HTMLResponse)
def interface():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(emp: Emprestimo):
    finalidades_validas = [
        "credit_card", "debt_consolidation", "educational",
        "home_improvement", "major_purchase", "small_business", "all_other"
    ]

    dados = {
        "politica_credito": emp.politica_credito,
        "taxa_juros": emp.taxa_juros,
        "parcela_mensal": emp.parcela_mensal,
        "log_renda_anual": emp.log_renda_anual,
        "relacao_divida_renda": emp.relacao_divida_renda,
        "pontuacao_fico": emp.pontuacao_fico,
        "dias_com_linha_credito": emp.dias_com_linha_credito,
        "saldo_rotativo": emp.saldo_rotativo,
        "utilizacao_rotativa": emp.utilizacao_rotativa,
        "consultas_ultimos_6meses": emp.consultas_ultimos_6meses,
        "atrasos_ultimos_2anos": emp.atrasos_ultimos_2anos,
        "registros_publicos": emp.registros_publicos,
    }

    for f in finalidades_validas:
        dados[f"finalidade_{f}"] = 1 if emp.finalidade == f else 0

    df = pd.DataFrame([dados])
    df_final = df.reindex(columns=colunas, fill_value=0)

    predicao = modelo.predict(df_final)[0]
    probabilidade = modelo.predict_proba(df_final)[0]

    return {
        "inadimplente": int(predicao),
        "resultado": "Alto risco de inadimplência" if predicao == 1 else "Baixo risco de inadimplência",
        "probabilidade_inadimplencia": round(float(probabilidade[1]), 4),
        "probabilidade_pagamento": round(float(probabilidade[0]), 4),
        "modelo": "RandomForestClassifier"
    }