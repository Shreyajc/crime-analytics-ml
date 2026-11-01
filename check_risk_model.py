import joblib

print("=" * 70)
print("RISK MODEL INSPECTION")
print("=" * 70)

model = joblib.load("crime_risk_model.pkl")

print("\nMODEL TYPE:")
print(type(model))

print("\nMODEL:")
print(model)

print("\nMODEL STEPS:")

if hasattr(model, "named_steps"):
    for name, step in model.named_steps.items():
        print(f"\n{name}:")
        print(step)

print("\nMODEL FEATURES:")

if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)

elif hasattr(model, "named_steps"):

    for name, step in model.named_steps.items():

        if hasattr(step, "feature_names_in_"):
            print(f"{name}:")
            print(step.feature_names_in_)

print("\nMODEL CLASSES:")

if hasattr(model, "classes_"):
    print(model.classes_)

elif hasattr(model, "named_steps"):

    for name, step in model.named_steps.items():

        if hasattr(step, "classes_"):
            print(f"{name}:")
            print(step.classes_)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)