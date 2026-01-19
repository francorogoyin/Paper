# Comparacion del paper con datos de los txt.

Este documento compara los valores reportados en el paper con los valores
obtenidos en los archivos txt de resultados.

- ✅ = Valor correcto o coincidente.
- ❌ = Valor incorrecto o discrepante.

---

## Parrafo 1 - Modelos GLM de autopercepciones.

### Texto del paper.

> "Each population showed a symbolic ideological left-wing to right-wing
> self-perception consistent with the original segregation criteria in
> these populations (Fig. 2Ai). The GLM model to explain this PI
> indicator for GE population [Deviance explained: 0.56; F-value: 220.75;
> AIC: 9573.91; BIC: -16649.61; p-value of the overall model: 1.11e-16]
> shows that some political variables (vote 2019, PASO_2023_category), as
> well as some written Media are significant independent variables
> (Supplementary Table 5). The Conservative&Progressive self-perception
> scale was also consistent but showed less discriminatory capacity
> (Fig. 2Aii), which could reflect that part of the population represented
> as Libertarian Right-wing does not perceive itself as entirely
> conservative. The GLM model to explain the Conservative&Progressive
> scale [Deviance explained: 0.259; F-value: 68.75; AIC: 11593.34;
> BIC: -11147.17; p-value of the overall model: 1.11e-16] shows also age,
> some argentinean regions, written Media and an online social network
> (facebook) as significant variables (Supplementary Table 6). The
> Peronism&anti-Peronism symbolic scale clearly showed two poles: one very
> anti-peronist (Moderate and Libertarian right-wing) and the other
> peronist (Progressivism). Interestingly, both the Left-wing and the
> Centre populations have a wide distribution with a neutral mean
> (Fig. 2Aiii). This GLM model [Deviance explained: 0.566; F-value: 179.07;
> AIC: 11360.17; BIC: -12002.08; p-value of the overall model: 1.11e-16]
> shows some political variables (vote 2019, PASO_2023_category), some
> written Media and some online social network as significant variables
> (Supplementary Table 7)."

### Comparacion - Modelo GLM Autopercepcion_Izq_Der.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Deviance | 0.56 | 0.5623 | ✅ |
| F-value | 220.75 | 222.3597 | ❌ |
| AIC | 9573.91 | 9656.32 | ❌ |
| BIC | -16649.61 | -16806.75 | ❌ |
| p-value | 1.11e-16 | 1.11E-16 | ✅ |

### Comparacion - Modelo GLM Autopercepcion_Con_Pro.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Deviance | 0.259 | 0.2575 | ❌ |
| F-value | 68.75 | 64.0332 | ❌ |
| AIC | 11593.34 | 11700.04 | ❌ |
| BIC | -11147.17 | -11223.47 | ❌ |
| p-value | 1.11e-16 | 1.11E-16 | ✅ |

### Comparacion - Modelo GLM Autopercepcion_Per_Antiper.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Deviance | 0.566 | 0.5636 | ❌ |
| F-value | 179.07 | 198.5489 | ❌ |
| AIC | 11360.17 | 11475.17 | ❌ |
| BIC | -12002.08 | -12054.54 | ❌ |
| p-value | 1.11e-16 | 1.11E-16 | ✅ |

### Variables significativas mencionadas.

**Autopercepcion_Izq_Der:**
- ✅ Variables politicas (vote 2019).
- ✅ Variables politicas (PASO 2023).
- ✅ Medios escritos.

**Autopercepcion_Con_Pro:**
- ✅ Edad.
- ✅ Regiones argentinas.
- ✅ Medios escritos.
- ✅ Red social (Facebook).

**Autopercepcion_Per_Antiper:**
- ✅ Variables politicas (vote 2019).
- ✅ Variables politicas (PASO 2023).
- ✅ Medios escritos.
- ✅ Red social.

---

## Parrafo 2 - Modelos GLM de indices ideologicos.

### Texto del paper.

> "When the populations were analysed using the progressivism and
> conservatism scales, the results were also consistent (Fig.2Bi; Bii).
> Given that both scales are constructed based on the response of
> independent items -despite belonging to the same instrument (42)-, the
> degree of redundancy of both scales was assessed by correlation. A
> significant correlation coefficient of -0.66 was observed, indicating
> that both scales are consistent with each other, but not completely
> redundant (Fig.2C). The GLM models for progressivism [Deviance explained:
> 0.609; F-value: 185.68; AIC: 3643.23; BIC:-21109.72; p-value: 1.11e-16]
> and conservativism indices [Deviance explained: 0.62; F-value: 217.25;
> AIC: 1291.47; BIC: -21624.16; p-value: 1.11e-16] evidence gender (male),
> e-social and political variables (vote 2019, PASO_2023_category,
> Peronist&Anti-Peronist self-perception, conservative cluster), and some
> written Media outlets as significant variables (Supplementary Table 8-9)."

### Comparacion - Modelo GLM Indice_Progresismo.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Deviance | 0.609 | 0.7017 | ❌ |
| F-value | 185.68 | 383.0595 | ❌ |
| AIC | 3643.23 | 2906.61 | ❌ |
| BIC | -21109.72 | -21499.70 | ❌ |
| p-value | 1.11e-16 | 1.11E-16 | ✅ |

### Comparacion - Modelo GLM Indice_Conservadurismo.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Deviance | 0.62 | 0.6365 | ❌ |
| F-value | 217.25 | 219.9594 | ❌ |
| AIC | 1291.47 | 1212.22 | ❌ |
| BIC | -21624.16 | -21823.78 | ❌ |
| p-value | 1.11e-16 | 1.11E-16 | ✅ |

### Correlacion entre escalas.

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Coeficiente | -0.66 | No encontrado | ❌ |

### Variables significativas mencionadas.

- ✅ Genero masculino (txt: Genero_Masculino).
- ✅ Variables politicas vote 2019 (txt: Voto_2019_Roberto_Lavagna).
- ✅ Categoria PASO 2023 (txt: Categoria_PASO_2023_*).
- ✅ Autopercepcion Peronismo vs anti-Peronismo (txt: Autopercepcion_Per_Antiper).
- ✅ Cluster conservador (txt: Conservative_Cluster).
- ✅ Medios escritos (txt: Medios_Prensa_*).
- ✅ Redes sociales (txt: Influencia_Redes o Red_Social_Youtube).

---

## Parrafo 3 - Analisis de clustering.

### Texto del paper.

> "Since the indices of conservatism and progressivism are calculated on
> the basis of responses to several items (Supplementary Table 2), the
> indices themselves do not allow us to discriminate between different
> conservative or progressive profiles. To explore whether different
> profiles existed, a clustering analysis was performed on the basis of
> each subject's responses for either the conservative or the progressive
> items. We observed 5 different clusters for the conservative dimension,
> and 6 for the progressive dimension. However, taking into account the
> number of subjects in each cluster, only 2 conservative and 3
> progressive clusters were taken into account for the analysis (Fig.2D),
> the remaining ones being assumed to be noisy clusters. Conservative
> cluster 1 seems to be composed of the most conservative population:
> opposing women's and children's rights (items 3, 8 and 19), indigenous
> and immigrants' rights (item 16 and 23), state participation in the
> economy (items 6, 9, 27 and 29), and health and responsible consumption
> policies (item 11); and in favour of increased restrictions on social
> protests (item 24) and even in favour of a military government (item 7).
> Progressive clusters (cluster 0 being less progressive and cluster 2
> more progressive) are mainly discriminated according to degrees of
> agreement with state involvement in the economy (items 6, 9, 20, 27 and
> 29), indigenous and immigrants' rights (item 16 and 23), with health
> and responsible consumption policies (item 11), and with social
> mobilisations (item 24). All this analysis suggests that there could be
> at least 4 qualitatively different profiles when analysing both scales
> (Fig.2Diii)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Clusters conservadores encontrados | 5 | No encontrado | ❌ |
| Clusters progresistas encontrados | 6 | No encontrado | ❌ |
| Clusters conservadores usados | 2 | No encontrado | ❌ |
| Clusters progresistas usados | 3 | No encontrado | ❌ |
| Perfiles cualitativos totales | 4 | No encontrado | ❌ |
| Descripcion por items | Menciona items | No encontrado | ❌ |

---

## Parrafo 4 - Modelos de cercania a candidatos.

### Texto del paper.

> "When assessing closeness to each candidate for each of the populations
> analysed, the results were also consistent with expectations and
> conservativism index (Fig.2E-F). The binary models constructed for
> closeness to each candidate (Supplementary Table 10) show different
> significant independent variables (and with different odds ratios),
> which reinforces the decision to analyse the populations separately.
> In general, as expected, different political variables end up explaining
> the variance of closeness to different candidates (Supplementary Table
> 11). In the case of the model for closeness to Massa, the positivity
> index was also significant. In fact, when comparing the means of the
> positivity index for each population, values below 17.5 were observed
> in all cases, demonstrating that a negative electoral climate prevailed.
> However, the progressive population showed less negativity than the
> other populations (Supplementary Figure 1D)."

### Comparacion - Modelos logisticos de cercania.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Variables distintas por candidato | Indica | Massa vs Milei | ✅ |
| OR distintos por candidato | Indica | Diferentes | ✅ |
| Variables politicas explican cercania | Indica | Categoria_PASO_2023_* y Voto_2019_* | ✅ |

### Cercania a Massa.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Indice_Positividad significativo | Indica | OR 1.164, p 2.28E-18 | ✅ |

### Indice_Positividad por poblacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Media menor a 17.5 en todos los casos | 17.5 | Max 16.4177 | ✅ |
| Progresismo menos negatividad | Indica | Progressivism 16.4177 mayor que otras | ✅ |

---

## Parrafo 5 - Analisis socioeconomico por poblacion.

### Texto del paper.

> "Further analysis was carried out to characterise these populations in
> terms of socio-economic variables. The results of General Election (to
> see results of Ballotage, see Supplementary Figure 2) show that in the
> Moderate right-wing (b) and Libertarian right-wing populations, men
> perceived themselves to be more right-wing than women in the same or
> other populations (Fig. 3Ai). In contrast to the other populations, the
> Libertarian population showed a lower mean age of men compared to women
> in the same population and in the other populations (Fig.3D). Although
> a significant proportion perceived themselves as having a medium to
> medium-low socio-economic level, similar to the left-wing and
> progressive population, the libertarian population had a higher
> proportion of subjects with a maximum level of secondary education
> (compared to the other populations) (Fig.3E). It is also the population
> that recognises a greater influence of social networks, especially
> networks with audiovisual content (Instagram, Facebook, Youtube, Tiktok)
> but also Whatsapps and Twitter (X) (Supplementary Figure 3). No
> differences were observed between the populations with respect to
> written Media influence, although greater consumption of Clarin, La
> Nacion and Infobae was observed in the right-wing populations, and
> Pagina12 in the progressive population, consistent with previous
> results (43)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Genero y autopercepcion derecha | Indica | No encontrado | ❌ |
| Edad por genero en libertaria | Indica | No encontrado | ❌ |
| Nivel educativo secundario mayor en libertaria | Indica | No encontrado | ❌ |
| Influencia de redes mayor en libertaria | Indica mayor | Libertaria 3.1459, max 3.7736 | ❌ |
| Influencia de prensa sin diferencias | Indica | Sin prueba | ❌ |
| Consumo de Clarin, La Nacion, Infobae y Pagina12 | Indica | No encontrado | ❌ |

---

## Parrafo 6 - Modelos robustos de OCI.

### Texto del paper.

> "To evaluate the change of rating of items with politicised content, we
> use the Opinion Change Indices (OCI) and their response time
> differences. In the first case, we assess the difference between the
> rating of items with no association to any candidate and when associated
> with a right-wing or left-wing candidate. Given the nature of the data,
> robust regression models were used here to assess which independent
> variables explain the variance of the opinion change indices
> (Supplementary Tables 12-15)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Uso de modelos robustos | Indica | Seccion MODELOS ROBUSTOS | ✅ |
| Indices OCI y tiempos | Indica | Usa CO_* y CT_* | ✅ |

---

## Parrafo 7 - Modelos robustos OCIcon y OCIpro.

### Texto del paper.

> "Robust regression models showed that ideological orientation is a key
> predictor of opinion change (Supplementary Tables 12-15). In the OCIcon
> model [Robust scale = 0.32; AIC = 4733.55; BIC=4828.43; p=6.93e-21],
> the Conservatism Index showed the greatest effect [Coefficient=-0.34;
> p=1.4e-21], indicating a negative and highly significant relationship:
> the more progressive, the lower the OCIcon score. Similarly, in the
> model of OCIpro [Robust scale = 0.39; AIC = 5275.59; BIC=5364.56;
> p=1.13e-26], the Progressivism Index [Coefficient=-0.3; p=1.25e-32] and
> Conservatism Index [Coefficient=-0.093; p=3.19e-3] had the most
> pronounced impact, showing that higher levels of conservatism are
> associated with a lower willingness to change on progressive issues.
> The analyses of Opinion Change disaggregated by association (OCIcon;r;
> OCIcon;l; OCIpro;r; OCIpro;l) confirmed these relationships with the
> conservatism and progressivism indices (Supplementary Tables 12-15).
> Overall, the results confirm that ideological orientations—progressivism
> and conservatism—exert a substantive and negative influence on the
> magnitude of opinion change, suggesting that such beliefs may function
> as cognitive anchors that limit variability in the monitoring of
> relevant information."

### Comparacion - OCIcon (CO_Con).

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Robust scale | 0.32 | 0.320343 | ✅ |
| AIC | 4733.55 | 4738.61 | ❌ |
| BIC | 4828.43 | 4821.64 | ❌ |
| p-value modelo | 6.93e-21 | 2.80E-20 | ❌ |
| Coef Indice_Conservadurismo | -0.34 | -0.3166 | ❌ |
| p Indice_Conservadurismo | 1.4e-21 | 1.66E-21 | ✅ |
| Relacion negativa | Indica | Beta negativa | ✅ |

### Comparacion - OCIpro (CO_Pro).

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Robust scale | 0.39 | 0.386611 | ✅ |
| AIC | 5275.59 | 5268.98 | ❌ |
| BIC | 5364.56 | 5363.88 | ❌ |
| p-value modelo | 1.13e-26 | 5.02E-28 | ❌ |
| Coef Indice_Progresismo | -0.3 | -0.3094 | ❌ |
| p Indice_Progresismo | 1.25e-32 | 1.07E-33 | ❌ |
| Coef Indice_Conservadurismo | -0.093 | -0.1310 | ❌ |
| p Indice_Conservadurismo | 3.19e-3 | 3.82E-04 | ❌ |
| Relacion negativa | Indica | Beta negativa | ✅ |

### OCI por asociacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| CO_Con_Der, CO_Con_Izq, CO_Pro_Der, CO_Pro_Izq | Confirmacion | Modelos existen | ✅ |

---

## Parrafo 8 - Modelos robustos por item.

### Texto del paper.

> "However, the general analysis has certain limitations. When the Opinion
> Change was analysed by item, differences were observed between items,
> suggesting that the change of ranting was dependent on the item´s
> politic content (Fig.4A). For this more comprehensive analysis, robust
> regression models were performed for items that showed significant
> differences between populations (Supplementary Table 13 and 15). For
> progressive items associated with left-wing candidates, the models
> showed that the Progressiveness Index variable had the greatest impact
> [item 24: β=-0.13 (p=3.9e-2); item 5: β=-0.24(p=2.48e-3); item 9:
> β=-0.25 (p=6.94e-5); item 25: β=-0.26 (p=2.0e-4)], suggesting that the
> more progressive the participant, the less change of opinion there is
> on these items associated with left-wing candidates. For conservative
> items associated with left-wing candidates, the models showed that the
> conservatism index variable had the greatest impact [item 8: β=-0.3
> (p=4.55e-4); item 30: β=-0.66 (p=2.82e-9); item 3: β=-0.28 (p=2.81e-3);
> item 10: β =-0.48 (p=2.99e-9)]; this was also the case when the items
> were associated with right-wing candidates [item 30: β =-0.67
> (p=3.29e-10); item 10: β =-0.64 (p=2.52e-8)]. These results suggest
> that the more conservative the participant, the less likely they are to
> change their opinion on these items associated with left-wing or
> right-wing candidates."

### Items progresivos con izquierda.

| Item | Paper (β, p) | txt (β, p) | Estado |
|------|--------------|------------|--------|
| Item 24 | -0.13, 3.9e-2 | -0.1252, 0.0390 | ✅ |
| Item 5 | -0.24, 2.48e-3 | -0.3269, 1.73E-04 | ❌ |
| Item 9 | -0.25, 6.94e-5 | -0.1953, 0.0177 | ❌ |
| Item 25 | -0.26, 2.0e-4 | -0.3657, 5.28E-06 | ❌ |

### Items conservadores con izquierda.

| Item | Paper (β, p) | txt (β, p) | Estado |
|------|--------------|------------|--------|
| Item 8 | -0.3, 4.55e-4 | -0.2665, 0.0019 | ❌ |
| Item 30 | -0.66, 2.82e-9 | -0.6285, 6.24E-09 | ❌ |
| Item 3 | -0.28, 2.81e-3 | Indice_Conservadurismo no aparece | ❌ |
| Item 10 | -0.48, 2.99e-9 | -0.5352, 1.49E-07 | ❌ |

### Items conservadores con derecha.

| Item | Paper (β, p) | txt (β, p) | Estado |
|------|--------------|------------|--------|
| Item 30 | -0.67, 3.29e-10 | -0.6650, 1.76E-09 | ❌ |
| Item 10 | -0.64, 2.52e-8 | -0.6667, 3.90E-08 | ❌ |

---

## Parrafo 9 - Magnitud de OCI por poblacion.

### Texto del paper.

> "To assess the magnitude of the opinion change, the difference between
> the opinion change associated with left-wing candidates and that
> associated with right-wing candidates was evaluated. This analysis was
> performed separately for progressive items (OCIpro) and conservative
> items (OCIcon). It was observed that left-wing and progressive
> populations showed a greater magnitude in the OCI for progressive items,
> while for right-wing populations, the greatest magnitude was seen for
> the OCI of conservative items (Fig. 4B). These results suggest that
> ideological extremes show similar patterns of opinion change associated
> with candidates, but that these patterns depend on the political content
> of the items."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Magnitud OCIpro mayor en izquierda/progresistas | Indica | No encontrado | ❌ |
| Magnitud OCIcon mayor en derecha | Indica | No encontrado | ❌ |
| Dependencia del contenido politico | Indica | No encontrado | ❌ |

---

## Parrafo 10 - Congruente vs incongruente.

### Texto del paper.

> "In order to assess whether the cognitive processes behind these changes
> of opinion are related to the congruence between the type of item
> (conservative or progressive) and the type of association (right-wing or
> left-wing candidate), congruent (OCICON: progressive items associated to
> left-wing candidates and conservative items associated to right-wing
> candidates) versus incongruent (OCIINC: progressive items associated to
> right-wing candidates and conservative items associated to left-wing
> candidates) changes of opinion were evaluated. In the OCICON model
> [Robust scale = 0.257; AIC = 4137.88; BIC=4232.8; p=9.49e-21], the
> Conservatism [Coefficient=-0.21; p=4.97e-17] and progressivism
> [Coefficient=-0.17; p=3.04e-18] indices showed the greatest effect,
> indicating a negative and highly significant relationship. In the
> OCIINC model [Robust scale = 0.24; AIC = 3919.96; BIC=3985.22;
> p=7.5e-19], also the Conservatism [Coefficient=-0.20; p=2.94e-13] and
> progressivism [Coefficient=-0.17; p=1.04e-21] indices showed the
> greatest effect. It should be noted that when the association is
> congruent, the change of opinion reflects greater agreement, while when
> the association is incongruent, the change of opinion reflects less
> agreement (Fig. 4Ci). In the first condition (OCICON), participants take
> significantly less time to respond than in the second condition (OCIINC)
> (Fig.4Cii). When these differences were analysed separately for each
> voter population, similar differences were observed, but not all
> populations showed significant differences (Fig. 4D). When comparing the
> patterns of opinion change between the two elections, some differences
> were observed. The magnitude of opinion change between the congruent and
> incongruent conditions is greater for the first election than for the
> second one (Fig. 4C-D), while response times showed a (non-significant)
> tendency to be shorter for the second election."

### Comparacion - OCICON (CO_Congruente).

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Robust scale | 0.257 | 0.256789 | ✅ |
| AIC | 4137.88 | 4136.70 | ❌ |
| BIC | 4232.8 | 4243.49 | ❌ |
| p-value modelo | 9.49e-21 | 2.77E-20 | ❌ |
| Coef Indice_Conservadurismo | -0.21 | -0.2359 | ❌ |
| p Indice_Conservadurismo | 4.97e-17 | 2.17E-16 | ❌ |
| Coef Indice_Progresismo | -0.17 | -0.1745 | ❌ |
| p Indice_Progresismo | 3.04e-18 | 1.02E-18 | ❌ |
| Relacion negativa | Indica | Betas negativas | ✅ |

### Comparacion - OCIINC (CO_Incongruente).

| Estadistico | Paper | txt | Estado |
|-------------|-------|-----|--------|
| Robust scale | 0.24 | 0.238514 | ✅ |
| AIC | 3919.96 | 3924.06 | ❌ |
| BIC | 3985.22 | 3989.31 | ❌ |
| p-value modelo | 7.5e-19 | 7.52E-18 | ❌ |
| Coef Indice_Conservadurismo | -0.20 | -0.1962 | ❌ |
| p Indice_Conservadurismo | 2.94e-13 | 1.21E-12 | ❌ |
| Coef Indice_Progresismo | -0.17 | -0.1732 | ❌ |
| p Indice_Progresismo | 1.04e-21 | 1.72E-21 | ❌ |
| Relacion negativa | Indica | Betas negativas | ✅ |

### Comparaciones adicionales.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Mayor acuerdo en congruente vs incongruente | Indica | Medias CO y CT con p significativas | ✅ |
| Diferencias por poblacion | No todas significativas | Parciales por categoria | ✅ |
| Mayor magnitud en primera eleccion | Indica | CO congruente 0.1138 vs 0.0482; CO incongruente -0.1025 vs -0.0320 | ✅ |
| Tiempos mas cortos en segunda eleccion | Tendencia no significativa | No reporta | ❌ |

---

## Parrafo 11 - Curvas balanceadas y Fig 4E.

### Texto del paper.

> "To assess the change of opinion when the association with a candidate
> is consistent with the ideological self-perception of the population,
> we analysed the change of opinion on conservative or progressive items
> in the left-wing population (such as the integration of left-wing and
> progressive populations) associated with a left-wing candidate, or in
> the right-wing population (moderate and libertarian right-wing)
> associated with a right-wing candidate. Here, a congruent condition
> occurs when, for the left-wing population, the change of opinion on
> progressive items is analysed, while for the right-wing population, it
> is analysed on conservative items. The inverse combination for each
> population (left-wing population – left-wing candidate – conservative
> items or right-wing population – right-wing candidate – progressive
> items) corresponds to incongruent conditions. This analysis was done
> using the rating of non-associated items as a reference. As the
> population per se is unbalanced, with a higher proportion of left-wing
> or progressive participants, a balanced mean value was calculated for
> the rating of each item. To do this, the same number of left-wing/
> progressive participants as right-wing participants (n=605) were
> randomly selected and the mean was calculated. In this way, the balanced
> curve reflects a population mean value for a population with an
> equivalent proportion of left-wing and right-wing participants. Figure
> 4E shows the opinion change curves based on this reference for both
> populations, comparing with centre population, under the associations
> described above, for the general elections (i) or the ballotage (ii).
> The rating of the centre's population was evaluated as the average of
> the association of both candidates, as no significant differences were
> found between the two associations. A quantification of the effect
> (Fig. 4Eiii) shows that, when the condition is congruent, both
> populations show greater agreement with affinity political content, with
> no differences observed between populations or elections. However, when
> the association is incongruent, the right-wing population shows a
> greater effect than the left-wing population [Fig.4Eiii: for General
> Election, p = 0.0312; for Ballotage, p = 0.064]. Both populations showed
> a greater range of opinion change compared to the centre population,
> suggesting that, in line with our hypothesis, ideological extremes are
> more susceptible to opinion change due to ideological association."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Analisis de curvas con balance n=605 | Indica | No encontrado | ❌ |
| Asociaciones sin diferencias en centro | Indica | No encontrado | ❌ |
| Cuantificacion Fig 4Eiii GE p=0.0312 | Indica | No encontrado | ❌ |
| Cuantificacion Fig 4Eiii Ballotage p=0.064 | Indica | No encontrado | ❌ |
| Mayor efecto derecha en incongruente | Indica | No encontrado | ❌ |

---

## Parrafo 12 - Interpretacion de resultados.

### Texto del paper.

> "All these results suggest that opinion change may be influenced not only
> by the ideological content of the item being evaluated, but also by its
> association with a political candidate who is consistent or inconsistent
> with that content. When the association is consistent, the process takes
> on the characteristics of a more automatic process (system 1), in line
> with the theoretical framework of cognitive shortcuts. Analysis between
> elections suggests that greater polarisation in the electoral offer does
> not necessarily lead to greater radicalisation of opinions."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Influencia de contenido ideologico y asociacion a candidato | Indica | No encontrado | ❌ |
| Proceso mas automatico (system 1) en congruente | Indica | No encontrado | ❌ |
| Polarizacion no implica radicalizacion | Indica | No encontrado | ❌ |

---

## Parrafo 13 - Objetivo del estudio.

### Texto del paper.

> "The aim of this study is to assess whether participants change their
> opinion regarding political statements when associated with opposing
> ideological references (candidates), and whether this opinion change
> depends on the ideological positioning of the participants."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Objetivo del estudio | Indica | No encontrado | ❌ |

---

## Parrafo 14 - Marco conceptual de ideologia.

### Texto del paper.

> "Political ideology has been conceptualised as an aspect of personality
> (31), a heuristic (5), or an intersubjective map of identification with
> certain beliefs systems (26). The association of political references
> may evoke these ideological beliefs, which, in turn, could act as an
> anchor for the judgement of such statements. These belief systems,
> totally or partially shared by groups of people, can contribute to the
> constitution of the political and ideological identities of those
> groups. These identities are fundamental in generating a sense of
> belonging, such that individuals and political parties conceive of
> themselves as representatives of social and cultural categories (44).
> Social Identity Theory suggests a tendency to evaluate the traits of
> members of other identity groups negatively (45), since membership
> generates a marked distinction between members of different groups (46)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Marco conceptual de ideologia | Indica | No encontrado | ❌ |

---

## Parrafo 15 - Enfoque metodologico general.

### Texto del paper.

> "To evaluate our objective, it was necessary to identify the different
> populations politically, assess the evaluation of these statements
> without anchoring (without any association with a political candidate),
> and then analyse how this judgement changes when associated with a
> left-wing or right-wing candidate. Given that references to the left or
> right may vary depending on the participant, our experimental design
> uses each participant's own references, rather than establishing fixed
> associations (which could lead to ideological bias on the part of the
> researchers)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Procedimiento general del estudio | Indica | No encontrado | ❌ |

---

## Parrafo 16 - Caracterizacion ideologica y clustering.

### Texto del paper.

> "Populations were characterized in terms of their political ideology,
> using symbolic self-perception and operational scales. Most ideological
> scales showed a consistent gradient associated with each population.
> However, they were insufficient to discriminate between left-wing and
> progressist populations, for which the Peronist&anti-Peronist scale was
> also useful. This scale proved to be irreducible to the others. The
> proximity to the candidates representing each force was also consistent
> with expectations. Clustering analysis based on responses to conservative
> and progressive items revealed the coexistence of three different
> populations. Using progressive items as a substrate, we identified three
> different clusters, while with conservative items, only two clusters.
> The progressive clusters differ mainly in their stance on social policies
> and state involvement in the economy, while the conservative clusters
> differ more in their stance on policies to expand social rights (notably
> abortion, sex education, immigrants) and, to a lesser extent, state
> involvement in the economy. Although the literature describes two
> conservative profiles (47), here we cannot discriminate between these
> two populations. The positions found in the conservative population seem
> to oppose mainly the social policies that were bastions of previous
> progressive governments."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Autopercepciones y escalas ideologicas | Indica | Valores distintos | ❌ |
| Peronismo vs anti-peronismo como escala irreductible | Indica | No encontrado | ❌ |
| Cercania a candidatos consistente con expectativas | Indica | No encontrado | ❌ |
| Clustering de progresistas y conservadores | Indica | No encontrado | ❌ |

---

## Parrafo 17 - Variables socioeconomicas y medios.

### Texto del paper.

> "When we analysed the different populations in terms of socio-economic
> variables, we found that the libertarian population had a lower average
> age, that men perceived themselves as more right-wing, and that social
> networks had a greater influence. These results are consistent with the
> literature on these new right-wing populations in other countries
> (48–51), or countries as Argentina (52, 53). These right-wing groups are
> opposed to social rights, particularly gender policies—which in our case
> can be clearly seen in the assessments of related elements—and defend
> economic liberalism, with minimal state intervention in the economy. At
> the same time, consumption of certain media outlets is a significant
> variable in symbolic and operational political ideology models. This is
> consistent with what was previously observed for the 2019 (1, 43).
> Although it is not possible to demonstrate a causal relationship between
> consumption of certain media outlets and the political ideology of the
> consumers the media may constitute channels for the dissemination of
> biased and partisan information."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Edad menor en libertarios y genero mas a la derecha | Indica | No encontrado | ❌ |
| Influencia de redes mayor en libertarios | Indica | No encontrado | ❌ |
| Medios significativos en modelos | Indica | Contiene Medios_Prensa_* | ✅ |
| Relacion causal medios-ideologia | Indica | No encontrado | ❌ |

---

## Parrafo 18 - Anclaje y diseno ecologico.

### Texto del paper.

> "Here we evaluate the anchoring effect on implicit information
> (associated with ideological beliefs) shared by a social group. In this
> case, it is very difficult to conceive an experimental design in which
> the treatment is anchoring (since it precedes the experimental instance).
> Here, we are not interested in evaluating whether or not information can
> act as an anchor for the judgement of other information, but specifically
> whether information incorporated into these belief systems could have an
> anchoring effect. To this end, our experimental design is ecological and
> uses precisely this a priori information as an explanatory variable for
> the phenomenon under study: the change of opinion."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Efecto de anclaje en informacion implicita | Indica | No encontrado | ❌ |

---

## Parrafo 19 - Hipotesis y resultados generales.

### Texto del paper.

> "Under the hypothesis that the most ideologically polarised populations
> tend to change opinion anchoring in their ideological beliefs, we analyse
> the change in assessment for each type of item and whether it is
> associated with a right-wing or left-wing candidate. Our results suggest
> that changes in opinion, evaluated as changes in the degree of agreement
> or disagreement with those political statements, are modulated by
> ideological identification, both of the participants and of the
> ideological references with which these statements are associated. When
> a general analysis was carried out, responses were very noisy. A more
> exhaustive analysis for each item, broken down by voter population,
> showed differences, allowing us to infer that these belief systems share
> common elements but not necessarily all elements."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Cambios modulados por identificacion ideologica | Indica | No encontrado | ❌ |
| Analisis por item con diferencias | Indica | No encontrado | ❌ |

---

## Parrafo 20 - Congruente vs incongruente (interpretacion).

### Texto del paper.

> "To characterise the cognitive processes beyond these changes of opinion,
> opinion changes were evaluated under two different conditions: congruent
> and incongruent. The first refers to the condition of progressive items
> associated with left-wing candidates and conservative items associated
> with right-wing candidates. This type of association is consistent with
> what is expected for both ideological belief systems. However, the
> incongruent condition presents an inconsistent situation: progressive
> items associated with right-wing candidates and conservative items
> associated with left-wing candidates. This condition is expected to
> generate a kind of cognitive dissonance (60) that affects the evaluation
> process at a metacognitive level, either by affecting its fluidity or
> response time. In the first condition, the OCI was positive, reflecting
> greater agreement, while response time was shorter."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| OCI positivo en congruente | Indica | Medias CO congruente > incongruente | ✅ |
| Tiempo de respuesta menor en congruente | Indica | No concluyente | ❌ |

---

## Parrafo 21 - Comparacion entre elecciones.

### Texto del paper.

> "When comparing the patterns of opinion change between the two elections,
> some differences were observed. Although both data sets are independent
> (they do not necessarily involve the same participants), this analysis
> allows us to explore how opinion change may vary in a context where five
> political forces compete (general elections) versus a context where only
> two political forces compete (second round of elections). We observed
> that the magnitude of opinion change between congruent and incongruent
> conditions is greater in the first election than in the second,
> suggesting that greater polarisation in the electoral offer does not
> necessarily lead to a radicalisation of opinions associated with both
> ideological extremes. Response times showed a (non-significant) tendency
> to be shorter in the second election. It is also important to note that
> the ballotage is the final election, so it could also be influencing
> expectations regarding the election outcome."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Mayor magnitud en primera eleccion | Indica | CO congruente/incongruente mayores | ✅ |
| Tiempos mas cortos en segunda eleccion | Indica | No reporta no significancia | ❌ |

---

## Parrafo 22 - Interpretacion teorica de anclaje.

### Texto del paper.

> "Our results suggest that political ideology can indeed act as an
> implicit anchor for judging information relevant to political decisions.
> On the one hand, it supports the idea that Political Ideology can act as
> an intersubjective map that allows for the recognition of political
> options related to belief systems shared by groups with a strong
> ideological identity. On the other hand, it is compatible with the idea
> that, at the cognitive level, it can act as a heuristic, favouring more
> automatic and rapid processes (system 1) and disadvantaging reflective
> processes (system 2)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Interpretacion teorica (heuristicas, anclaje) | Indica | No encontrado | ❌ |

---

## Parrafo 23 - Implicancias y politicas.

### Texto del paper.

> "Although this work has several limitations that prevent us from delving
> deeper into these speculations, we propose the challenge of providing an
> interpretation of how the belief systems that sustain ideologies
> undermine people's degrees of freedom, making their decisions less
> reflective and more automatic. We believe that policies aimed at
> fostering greater critical and scientific thinking among the general
> population could allow for greater degrees of freedom without sacrificing
> diversity of opinion, which would lead to better democratic systems."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Implicancias sobre pensamiento critico | Indica | No encontrado | ❌ |

---

## Parrafo 24 - Limitaciones y perspectivas.

### Texto del paper.

> "LIMITATIONS AND PERSPECTIVES"

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Encabezado de limitaciones | Indica | No encontrado | ❌ |

---

## Parrafo 25 - Limitacion de muestreo.

### Texto del paper.

> "The main limitation of this study is the population sampling, as
> right-wing and left-wing/progressive populations are not equally
> represented in the data. This may be due to the fact that the former has
> anti-science beliefs, which discourages participation in experiments.
> However, given that the n populations were large, this allows for
> sufficient data collection from the former population (as well as the
> centrist population) to carry out the analyses proposed in the paper."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Limitacion por muestreo y n poblaciones | Indica | No encontrado | ❌ |

---

## Parrafo 26-27 - Leyendas y Figura 1.

### Texto del paper.

> "Figure 1. Political Scenario of 2023 Argentine Presidential Elections.
> Schematic of the experimental design (A). Electoral results of the
> General Election (B), and of the Ballotage (C). Sankey graph showing the
> electoral preference associated with the vote in the 2019 elections, the
> PASO elections in 2023 and the political closeness to each candidate in
> the General Election (D) and Ballotage (G). Political perception of each
> candidate on the symbolic scales of left-wing/right-wing and
> conservatism-progressivism for the General Elections (F) and
> Ballotage (G)."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Descripcion de escenario y diseno | Indica | No encontrado | ❌ |

---

## Parrafo 28 - Figura 2.

### Texto del paper.

> "Figure 2. Political Characterization of voting populations. A. Symbolic
> political characterization: i) Left-wing&Right-wing self-perception for
> each population (Kruskal-Wallis: p-value =1.53e-187); ii). Conservative&
> Progressive self-perception for each population (Kruskal-Wallis: p-value
> =2.36e-65); iii) Peronist/anti-peronist self-perception for each
> population (Kruskal-Wallis: p-value =7.87e-221). B. Operational political
> characterization: i) Progressivism scale for each population
> (Kruskal-Wallis: p-value =5.27e-206); ii) Conservatism scale for each
> population (Kruskal-Wallis: p-value =8.74e-194). C. Correlation between
> Progressivism and Conservatism indexes. D. Cluster analysis from median
> conservative or progressive items. E. Political closeness to: i) Bregman:
> Kruskal-Wallis: p-value =8.37e-189; ii) Massa: Kruskal-Wallis: p-value
> =1.03e-226; iii) Schiaretti: Kruskal-Wallis: p-value =9.30e-81; iv)
> Bullrich: Kruskal-Wallis: p-value =2.92e-230; iii) Milei: Kruskal-Wallis:
> p-value =3.34e-270. In all cases, post-hoc Dunn Test reveals by default
> significant differences (p-value < 0.001) in all comparations excepts
> when it was indicated (* p-value < 0.05; ** p-value < 0.01) or for those
> indicated as non-significant (ns). F. Correlation between Conservatism
> index and each candidate´s closeness."

### Comparacion - Tests Kruskal-Wallis.

| Variable | Paper (p-value) | txt (p-value) | Estado |
|----------|-----------------|---------------|--------|
| Autopercepcion Izq-Der | 1.53e-187 | 2.41E-190 | ❌ |
| Autopercepcion Con-Pro | 2.36e-65 | 6.25E-66 | ❌ |
| Autopercepcion Per-Antiper | 7.87e-221 | 1.28E-222 | ❌ |
| Indice Progresismo | 5.27e-206 | 5.31E-222 | ❌ |
| Indice Conservadurismo | 8.74e-194 | 1.94E-208 | ❌ |
| Cercania Bregman | 8.37e-189 | 1.61E-190 | ❌ |
| Cercania Massa | 1.03e-226 | 7.24E-230 | ❌ |
| Cercania Schiaretti | 9.30e-81 | 6.86E-81 | ❌ |
| Cercania Bullrich | 2.92e-230 | 9.15E-233 | ❌ |
| Cercania Milei | 3.34e-270 | 3.06E-273 | ❌ |

### Otros datos Figura 2.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Dunn post-hoc por defecto | Indica | No encontrado | ❌ |
| Correlacion progresismo-conservadurismo | Indica | No encontrado | ❌ |

---

## Parrafo 29 - Figura 3.

### Texto del paper.

> "Figure 3. Social Characterization of voting populations. A. Symbolic
> political characterization disaggregated by Gender and population: i)
> Left-wing&Right-wing self-perception; ii). Conservative&progressive
> self-perception; iii) Peronist&anti-peronist self-perception. B.
> Operational political characterization: i) Progressivism scale; ii)
> Conservatism scale. C. Closeness to candidates disaggregated by Gender.
> D. Age (white line represent the median) of population disaggregated by
> Gender and populations. E. Self-perceived socio-economic status (i) and
> maximus education level (ii) disaggregated by populations. In all cases,
> Mann-Whitney U Test (two-sided) significant results are shown in orange
> for female comparations, cyan for male comparations and black for
> comparisons between females and males within the same population:
> * p-value < 0.05; ** p-value < 0.01; *** p-value < 0.001."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Mann-Whitney por genero y poblacion | Indica | No encontrado | ❌ |

---

## Parrafo 30 - Figura 4.

### Texto del paper.

> "Figure 4. Opinion Change of items. A. Opinion Change of significant
> items desegregated by populations: the diamonds represent Opinion Change
> of items associated with right-wing candidates, while the circles
> represent left-wing candidates' association. Significant differences
> between left-wing & right-wing association were assessed using the
> Wilcoxon test; and between populations, Dunn Test with Bonferroni
> correction: * p-value < 0.05; ** p-value < 0.01; *** p-value < 0.001.
> B. Differences between the rates of opinion change associated with
> left-wing and right-wing candidates. C. Congruent & incongruent opinion
> change for the general election and ballotage. D. Opinion change
> disaggregated by voter population in each election. E. Opinion Change
> of left-wing (integrating left-wing and progressive populations),
> right-wing populations (integrating moderate and libertarian right-wing
> populations) and Centre population, respect of general rating. To obtain
> a balanced general rating, the right-wing population was combined with a
> number of left-wing participants (chosen randomly) equivalent to the
> right-wing population (for GE: n=605; for Ballotage: n=254). The
> significance between different populations and elections in
> quantification (iii) was assessed using the Mann-Whitney test, while the
> difference from zero was assessed using the Wilcoxon signed-rank test:
> * p-value < 0.05; ** p-value < 0.01; *** p-value < 0.001."

### Comparacion.

| Dato | Paper | txt | Estado |
|------|-------|-----|--------|
| Wilcoxon Izq vs Der por item | Indica | Tiene seccion Wilcoxon | ✅ |
| Dunn con Bonferroni entre poblaciones | Indica | No encontrado | ❌ |
| Mann-Whitney en cuantificacion | Indica | No encontrado | ❌ |
| n=605, n=254, n=84, n=43 | Indica | No encontrado | ❌ |

---

## Resumen de comparacion.

### Modelos GLM de autopercepciones.

| Modelo | Correctos | Incorrectos |
|--------|-----------|-------------|
| Autopercepcion_Izq_Der | 2 | 3 |
| Autopercepcion_Con_Pro | 1 | 4 |
| Autopercepcion_Per_Antiper | 1 | 4 |

### Modelos GLM de indices.

| Modelo | Correctos | Incorrectos |
|--------|-----------|-------------|
| Indice_Progresismo | 1 | 4 |
| Indice_Conservadurismo | 1 | 4 |

### Modelos robustos.

| Modelo | Correctos | Incorrectos |
|--------|-----------|-------------|
| OCIcon (CO_Con) | 4 | 3 |
| OCIpro (CO_Pro) | 3 | 6 |
| OCICON (CO_Congruente) | 3 | 6 |
| OCIINC (CO_Incongruente) | 3 | 6 |

### Datos no encontrados en txt.

- Analisis de clustering completo.
- Analisis socioeconomico detallado.
- Curvas balanceadas Fig 4E.
- Interpretaciones teoricas.
- Mann-Whitney y Dunn post-hoc detallados.