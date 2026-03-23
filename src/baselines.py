import load_data  # importerer load_data for å få tilgang til subset og day_cols
import preprocessing  # importerer preprocessing for å få tilgang til X_test og y_test

y_pred = [x_window[-7:] for x_window in preprocessing.X_test]  # bruker de siste 7 dagene i input som prediksjon

print("Antall prediksjoner:", len(y_pred))
print("Første prediksjon:", y_pred[0])
print("Første faktiske target:", preprocessing.y_test[0])