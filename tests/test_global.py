"""Tests globaux pour l'ensemble du projet MF_Tick.

Ce script permet de lancer tous les tests du projet de manière organisée.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_tests():
    """Lance tous les tests du projet."""
    import subprocess

    print("🚀 Lancement de tous les tests du projet MF_Tick")
    print("=" * 60)

    # Liste des répertoires de tests à exécuter
    test_dirs = [
        "tests/arima",
        "tests/data_cleaning",
        "tests/data_fetching",
        "tests/data_preparation",
        "tests/deep_learning",
        "tests/evaluation",
        "tests/garch",
        "tests/model",
        "tests/optimisation",
        "tests/training",
        "tests/utils",
    ]

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"\n📂 Exécution des tests dans {test_dir}")
            print("-" * 40)

            try:
                # Lancer pytest pour ce répertoire
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test_dir,
                    "-v", "--tb=short", "--color=yes"
                ], capture_output=True, text=True, cwd=project_root)

                # Analyser les résultats
                output_lines = result.stdout.split('\n')
                summary_line = ""
                for line in output_lines:
                    if "passed" in line and "failed" in line:
                        summary_line = line
                        break

                if result.returncode == 0:
                    print("✅" + " PASSED")
                else:
                    print("❌" + " FAILED")
                    # Afficher les erreurs importantes
                    for line in output_lines[-10:]:  # Dernières 10 lignes
                        if line.strip() and ("FAILED" in line or "ERROR" in line):
                            print(f"   {line}")

                # Extraire les statistiques si disponibles
                if summary_line:
                    print(f"   {summary_line}")

            except Exception as e:
                print(f"❌ Erreur lors de l'exécution: {e}")

        else:
            print(f"⚠️  Répertoire {test_dir} non trouvé, ignoré")

    print("\n" + "=" * 60)
    print("🏁 Exécution des tests terminée")

    # Résumé final
    print("\n📊 Pour plus de détails, lancez individuellement:")
    print("   python -m pytest tests/arima -v")
    print("   python -m pytest tests/model -v")
    print("   python -m pytest tests/garch -v")
    print("   etc.")

if __name__ == "__main__":
    run_tests()
