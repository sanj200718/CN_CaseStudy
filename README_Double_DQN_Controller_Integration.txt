DOUBLE DQN CONTROLLER INTEGRATION

Purpose
- Add a separate controller routing mode named `dqn_double`.
- Preserve the existing `dqn`, `dijkstra`, and `ospf` modes.
- Load routing_dqn_visited_double_candidate.pt using the 44-feature visited-aware inference wrapper.
- Keep the existing explicit Dijkstra fallback for dead ends.
- Do not modify env.py, routing_dqn.pt, or learning_curve.png.

Files in this package
- controller.py: adds separate loading/selection/status reporting for the Double DQN model.
- routing_core.py: supports the dqn_double mode and aliases.
- inference_visited_aware.py: the 44-feature inference wrapper used by the Double DQN checkpoint.
- tests/test_routing_core.py: adds tests for the new mode.
- smoke_test_double_dqn.py: verifies that the checkpoint loads and produces a legal next hop.

Before installing
The checkpoint itself is not included. The project root must already contain:
  routing_dqn_visited_double_candidate.pt

Recommended install steps (run one command at a time in the project root)
1. Back up the files that will be replaced:
   mkdir -p ~/Downloads/CN_CaseStudy-backup-before-double-dqn-integration
   cp controller.py ~/Downloads/CN_CaseStudy-backup-before-double-dqn-integration/controller.py
   cp routing_core.py ~/Downloads/CN_CaseStudy-backup-before-double-dqn-integration/routing_core.py
   cp inference_visited_aware.py ~/Downloads/CN_CaseStudy-backup-before-double-dqn-integration/inference_visited_aware.py
   cp tests/test_routing_core.py ~/Downloads/CN_CaseStudy-backup-before-double-dqn-integration/test_routing_core.py
2. Extract this ZIP into the project root. The ZIP includes paths for tests/:
   unzip -o ~/Downloads/double_dqn_controller_integration.zip -d ~/Downloads/CN_CaseStudy-network-setup
3. Validate syntax:
   python -m py_compile controller.py routing_core.py inference_visited_aware.py smoke_test_double_dqn.py tests/test_routing_core.py
4. Run unit tests:
   python -m unittest discover -s tests -v
5. Check the candidate checkpoint and 44-feature inference path:
   python smoke_test_double_dqn.py

Runtime notes
- The controller loads original and Double DQN checkpoints independently. If the candidate is missing or fails to load, existing modes remain available and `dqn_double` is rejected with an explanatory error.
- In controller REST API calls, use POST /routing/mode with JSON body {"mode":"dqn_double"} to select the candidate.
- Example (when the Ryu controller is running locally on port 8080):
  curl -s -X POST http://127.0.0.1:8080/routing/mode -H 'Content-Type: application/json' -d '{"mode":"dqn_double"}'
  curl -s http://127.0.0.1:8080/routing/status
- The status JSON reports `models_loaded.dqn_double` and `active_model_loaded`.
- The controller keeps Dijkstra fallback enabled. The live route-decision log records whether fallback was used.
- Run actual Ryu/Mininet tests in the Linux/WSL environment where Ryu and Mininet are installed; Mac unit tests do not establish end-to-end network performance.
- Do not replace routing_dqn.pt with the 44-feature checkpoint. They use different input dimensions.
