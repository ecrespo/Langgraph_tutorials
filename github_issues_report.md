
# GitHub Issues Health Report

---

## 1. Estado del Backlog

| Métrica | Valor |
|---|---|
| Total issues | 5,000 |
| Issues abiertos | 1,568 (31.4%) |
| Issues cerrados | 3,432 (68.6%) |
| Abiertos sin comentarios | 36 (2.3% de los abiertos) |

---

## 2. Velocidad del Equipo

| Métrica | Valor |
|---|---|
| Tiempo mediano de resolución | 60 días |
| Cerrados en < 7 días (respuesta rápida) | 186 (5.4%) |
| Abiertos por más de 30 días (acumulación) | 1,568 (100.0%) |

---

## 3. Distribución de Trabajo

**Top 3 tipos de issues:**

| Tipo | Count | % |
|---|---|---|
| feature | 2,730 | 54.6% |
| bug | 1,788 | 35.8% |
| question | 482 | 9.6% |

- **Label más frecuente:** `bug`
- **Issue más comentado:** "New flag --v1.0 for CLI tool..." (45 comentarios)

---

## 4. Deuda Técnica Estimada

| Métrica | Valor |
|---|---|
| Bugs abiertos | 571 |
| Issues de performance abiertos | 120 |
| Índice de deuda técnica | 13.8% |

✅ **Deuda técnica bajo control**

---

## 5. Duplicados Detectados

| Métrica | Valor |
|---|---|
| Pares potencialmente duplicados | 423,869 |
| Issues evitables (estimado 80%) | 339,095 |

**Top 3 pares más similares:**

1. Score `1.000` — *"server does not work as expected — am I missi..."* ≈ *"server does not work as expected — am I missi..."*
2. Score `1.000` — *"server does not work as expected — am I missi..."* ≈ *"server does not work as expected — am I missi..."*
3. Score `1.000` — *"server does not work as expected — am I missi..."* ≈ *"server does not work as expected — am I missi..."*


---

## 6. Recomendaciones

1. **Atender issues sin respuesta**: 36 issues abiertos tienen 0 comentarios
   (2% del backlog abierto). Asignar triaje semanal reduciría
   el tiempo de primera respuesta.

2. **Reducir acumulación de largo plazo**: 1,568 issues llevan más de 30 días
   abiertos (100%). Revisar si son candidatos a cerrar como `wontfix`
   o reclasificar como `backlog`.

3. **Implementar detección de duplicados**: Se detectaron 423,869 pares
   similares. Agregar una búsqueda automática de issues similares al crear uno nuevo
   podría evitar ~339,095 issues redundantes.

---
*Generado automáticamente por CodeAct Agent — LangGraph Patrón 10*
