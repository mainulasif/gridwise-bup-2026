# 3-Minute Architecture / Solution Video Script

Target length: about 2:30-2:50.

## 0:00-0:20 - Problem

"This is our GridWise solution for the BUP CSE Fest 2026 preliminary. The API receives 24 hours of campus demand, solar availability, electricity tariffs, battery parameters, and one to three natural-language operator notes. The goal is to understand those notes, apply every valid directive, return a valid 24-hour schedule, and minimize grid electricity cost."

## 0:20-0:55 - Architecture

Show the README architecture diagram.

"Our pipeline has four stages. First, a local language-capable generative model, Google FLAN-T5 Small, interprets each operator note and proposes structured directive JSON. Second, deterministic guardrails validate and normalize that untrusted output. Third, a linear optimizer applies the directives and computes the least-cost schedule. Fourth, an independent replay validator checks every hour before the API returns anything."

## 0:55-1:25 - LLM and guardrails

Show app/interpreter.py.

"The LLM is directly in the operator-note interpretation path. The guardrail layer enforces the official directive enum, note mapping, applies semantics, whole-hour windows, solar factors, battery reserve limits, and grid caps. Invalid model output fails safely instead of inventing a constraint."

## 1:25-1:55 - Optimizer

Show app/optimizer.py.

"The optimizer uses SciPy HiGHS. We use one signed battery-flow variable per hour: positive means discharge and negative means charge. This makes simultaneous charge and discharge impossible without a slower mixed-integer model. We enforce energy balance, effective solar, battery bounds, rate limits, directive windows, grid caps, and final battery neutrality, then minimize tariff-weighted grid import."

## 1:55-2:15 - Verification

"After optimization, we replay the schedule hour by hour and independently verify battery transitions, rate limits, solar use, energy balance, directive constraints, and end-of-day neutrality. Totals are recalculated from the returned plan."

## 2:15-2:40 - Run and test

"The service exposes exactly GET slash health and POST slash optimize-energy. It is deployed as one Railway service and also has a Docker fallback image. The repository contains public-sample validation and local tests."

## 2:40-2:50 - Finish

"The result is a reproducible LLM-to-guardrail-to-optimizer pipeline designed first for correctness and then for cost."
