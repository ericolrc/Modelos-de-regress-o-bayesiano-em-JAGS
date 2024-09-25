## Pacotes -------------------------------------------------------------------
library(R2jags)   # Para modelagem bayesiana com JAGS
library(coda)     # Para análise de cadeias MCMC
library(bayesplot) # Para visualização de resultados bayesianos

## Modelo de regressão Normal Generalizada -----------------------------------

# Código do modelo JAGS
model_code <- "
data {
   for (j in 1:n) {
        Zeros[j] <- 0 # Inicializa o vetor Zeros
  }
}
model {
  # Likelihood
  C <- 10000 # Constante para ajustar a média
  for (i in 1:n) {
    Zeros[i] ~ dpois(Zeros_mean[i]) # Distribuição de Poisson
    Zeros_mean[i] <- -L[i] + C # Média do modelo
    L[i] <- log(beta) - (1/alpha^beta) * (abs(y[i] - mu[i]))^beta - log(2 * alpha) - loggam(1/beta) 
    mu[i] <- b[2] * x1[i] + b[3] * x2[i] + b[4] * x3[i] # Previsão da média

    res[i] <- y[i] - mu[i] # Resíduos do modelo
  }

  # Priors para os coeficientes de regressão
  for (j in 2:4) {
    b[j] ~ dnorm(0, 0.01) # Distribuição normal com média 0 e variância 100
  }

  # Prior para o parâmetro da distribuição de cauda pesada
  beta <- 2 # Valor fixo para a cauda pesada 

  # Prior para alpha
  alpha ~ dgamma(0.10, 0.10) # Distribuição gamma para alpha
}
"

# Dados para o modelo
data = data.frame(
  y = c(250, 150, 128, 134, 110, 131, 98, 84, 147, 124, 128, 124, 147, 90, 96, 120, 102, 84, 86, 84, 134,
        128, 102, 131, 84, 110, 72, 124, 132, 137, 110, 86, 81, 128, 124, 94, 74, 89),
  cerebro = c(81.69, 103.84, 96.54, 95.15, 92.88, 99.13, 
              85.43, 90.49, 95.55, 83.39, 107.95, 92.41, 
              85.65, 87.89, 86.54, 85.22, 94.51, 80.80, 
              88.91, 90.59, 79.06, 95.50, 83.18, 93.55,
              79.86, 106.25, 79.35, 86.67, 85.78, 94.96,
              99.79, 88.00, 83.43, 94.81, 94.94, 89.40,
              93.00, 93.59),
  altura = c(64.5, 73.3, 68.8, 65.0, 69.0, 64.5, 66.0, 
             66.3, 68.8, 64.5, 70.0, 69.0, 70.5, 66.0, 
             68.0, 68.5, 73.5, 66.3, 70.0, 76.5, 62.0, 68.0, 
             63.0, 72.0, 68.0, 77.0, 63.0, 66.5, 62.5, 67.0, 
             75.5, 69.0, 66.5, 66.5, 70.5, 64.5, 74.0, 75.5),
  peso = c(118, 143, 172, 147, 146, 138, 175, 134, 172, 118, 
           151, 155, 155, 146, 135, 127, 178, 136, 180, 186, 
           122, 132, 114, 171, 140, 187, 106, 159, 127, 191,
           192, 181, 143, 153, 144, 139, 148, 179)
)

# Preparando os dados para o modelo JAGS
model_data <- list(y = data[,1], x1 = data[,2], x2 = data[,3], x3 = data[,4],
                   n = nrow(data))

# Parâmetros a serem salvos do modelo
model_parameters <- c("b", "alpha")

# Executando o modelo JAGS
jags_model <- jags(data = model_data,
                   parameters.to.save = model_parameters,
                   model.file = textConnection(model_code),
                   n.chains = 3,
                   n.iter = 100000,
                   n.burnin = 55000,
                   n.thin = 100, progress.bar = "text",
                   DIC = FALSE)

# Resultados do modelo
print(jags_model)

## Diagnóstico do modelo ---------------------------------------------------
jags_mcmc <- as.mcmc(jags_model) # Convertendo os resultados para MCMC
mcmc_acf(jags_mcmc, lags = 50) # ACF plot
mcmc_trace(jags_mcmc) # Trace plot
mcmc_dens(jags_mcmc) # Density plot

## Modelo de Distribuição Floor --------------------------------------------

# Código do segundo modelo JAGS
model_code <- "
data {
   for (j in 1:n) {
        Zeros[j] <- 0 # Inicializa o vetor Zeros
  }
}
model {
  # Likelihood
  C <- 10000 # Constante para ajustar a média
  for (i in 1:n) {
    Zeros_mean[i] <- -L[i] + C # Média do modelo
    Zeros[i] ~ dpois(Zeros_mean[i]) # Distribuição de Poisson
    L[i] <- -log(theta) - a * log(x[i] / theta) + round(log(x[i] / theta) - 0.5) # Cálculo de L
  }
  
  # Priors
  theta ~ dgamma(3, 8) # Distribuição gamma para theta
  a <- 5 # Valor fixo para a
  p[1] <- step(theta - 1.93)  # Cálculo da probabilidade
}
"

# Dados para o segundo modelo
amostra <- list(x = c(1, 1, 2, 4, 5), n = 5)

# Parâmetros a serem salvos do segundo modelo
model_parameters <- c("theta")

# Função para inicializar o modelo JAGS
jags.inits <- function() {
  list("theta" = 0.05) # Inicialização para theta
}

# Executando o segundo modelo JAGS
jags_model <- jags(data = amostra,
                   parameters.to.save = model_parameters,
                   model.file = textConnection(model_code),
                   n.chains = 3,
                   n.iter = 25000,
                   n.burnin = 15000,
                   n.thin = 10, progress.bar = "text",
                   DIC = FALSE)

# Resultados do segundo modelo
print(jags_model)

## Diagnóstico do segundo modelo -------------------------------------------
acfplot(as.mcmc(jags_model), lag.max = 50) # ACF plot
HPDinterval(as.mcmc(jags_model)) # Intervalo HPD
densityplot(as.mcmc(jags_model)) # Density plot
xyplot(as.mcmc(jags_model)) # XY plot

# Definindo a função a ser otimizada
funcao_otimizacao <- function(par, x) {
  theta <- par[1]
  n <- length(x)
  resultado <- -n * log(theta) - 5 * sum(log(x / theta)) + sum(floor(log(x / theta)))
  return(-resultado) # Retorna o valor negativo da função a ser otimizada
}

# Dados de exemplo para otimização
x <- c(1, 1, 2, 4, 5)
a <- 5  # Ajuste o valor de 'a' conforme necessário

# Chamada da função de otimização
resultado_otimizacao <- optim(par = c(9.651652e-54), fn = funcao_otimizacao, x = x)

# Resultados da otimização
resultado_otimizacao$par
