# Codex working rules — Dream Labs Hackathon

Read `README.md` first. The onsite brief asks for a tool for Dream-1 users, built with the Dream Labs open-source library and tested in simulation.

## Operating rules

1. Separate **CONFIRMED**, **HYPOTHESIS**, and **UNKNOWN** in technical plans.
2. Do not hallucinate SDK packages, API endpoints, JSON schemas, authentication, simulator APIs, robot hardware, model capabilities, or competition rules.
3. Prefer official challenge documents and official developer documentation over secondary sources. If unavailable, ask for the relevant URL or document.
4. Before changing code: summarize objective, affected files, assumptions and how to verify success.
5. Make small, reversible changes. Preserve a working baseline and avoid unnecessary architecture.
6. Keep credentials in environment variables; never commit secrets, personal data, or private competition assets.
7. Provide unit tests using mocks for any integration that lacks confirmed SDK documentation.
8. Do not download models, install large simulators, incur paid API costs, or trigger remote actions without explicit approval.
9. Use type hints and clear units/coordinate-frame naming. Validate action bounds and error handling once schemas are known.
10. Track experiments with seed, config, version, metric, and failure cases when the official task permits.
11. Respect organizer rules about pre-existing code, third-party tools, publication, and generated code.
12. When the challenge brief arrives, first update requirements and identify the smallest end-to-end official baseline. Only then implement challenge-specific functionality.

