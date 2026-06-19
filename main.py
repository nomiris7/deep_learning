import numpy as np
from llm_generator import generate_architecture_from_llm
from encoder import encode_architecture
from cnn_builder import train_and_evaluate_cnn
from surrogates import get_xgboost_surrogate, get_mlp_surrogate, evaluate_surrogate


def main():
    print("--- Start NAS Pipeline ---")

    # 1. Inicjalizacja początkowego zbioru danych (Pusty na start, w rzeczywistości tu ładujemy wygenerowane wcześniej np. 50 sztuk)
    dataset_X = []
    dataset_y = []

    print("Inicjalizacja początkowego zbioru (symulacja 20 architektur)...")
    for _ in range(20):
        arch = generate_architecture_from_llm()
        if arch:
            acc = train_and_evaluate_cnn(arch)
            dataset_X.append(encode_architecture(arch))
            dataset_y.append(acc)

    best_accuracy = max(dataset_y) if dataset_y else 0.0

    # 2. Inicjalizacja i nauka Surogatu
    surrogate = get_xgboost_surrogate()  # lub get_mlp_surrogate()
    surrogate.fit(np.array(dataset_X), np.array(dataset_y))

    # 3. Iteracyjna pętla (np. 10 iteracji NAS)
    num_iterations = 10

    for i in range(num_iterations):
        print(f"\n--- Iteracja {i + 1}/{num_iterations} ---")

        # Krok 1: LLM
        new_arch = generate_architecture_from_llm()
        if not new_arch:
            continue

        # Krok 2: Kodowanie
        encoded_vec = encode_architecture(new_arch)

        # Krok 3: Predykcja
        pred_acc = surrogate.predict(encoded_vec.reshape(1, -1))[0]
        print(
            f"LLM wygenerował architekturę. Przewidywane Acc: {pred_acc:.2f}% (Obecny Best: {best_accuracy:.2f}%)"
        )
        margin = 5.0 * (1 - (i / num_iterations))
        # Krok 4 & 5: Trenuj w pełni jeśli obiecująca
        if pred_acc > (best_accuracy - margin):
            print("Architektura obiecująca! Właściwy trening na CIFAR-10...")
            real_acc = train_and_evaluate_cnn(new_arch)
            print(f"Rzeczywiste Accuracy: {real_acc:.2f}%")

            dataset_X.append(encoded_vec)
            dataset_y.append(real_acc)

            if real_acc > best_accuracy:
                best_accuracy = real_acc

            # Krok 6: Douczanie surogatu
            surrogate.fit(np.array(dataset_X), np.array(dataset_y))

            # Ewaluacja surogatu na zebranym dotąd zbiorze
            preds = surrogate.predict(np.array(dataset_X))
            metrics = evaluate_surrogate(dataset_y, preds)
            print(
                f"Metryki surogatu na aktualnym zbiorze: MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.2f}"
            )
        else:
            print("Architektura odrzucona przez model surogatowy.")


if __name__ == "__main__":
    main()
