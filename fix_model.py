import h5py
import json
import os

def fix_model(filename):
    print(f"\n🔧 Fixing: {filename}")

    if not os.path.exists(filename):
        print("❌ File not found!")
        return

    with h5py.File(filename, "r+") as f:

        model_config_attr = f.attrs.get("model_config")

        if model_config_attr is None:
            print("❌ No model_config found!")
            return

        # ✅ HANDLE BOTH STRING & BYTES
        if isinstance(model_config_attr, bytes):
            model_config_str = model_config_attr.decode("utf-8")
        else:
            model_config_str = model_config_attr

        config = json.loads(model_config_str)

        # 🔁 PATCH FUNCTION
        def fix(obj):
            if isinstance(obj, dict):

                if "config" in obj:
                    layer_config = obj["config"]

                    # ✅ Fix batch_shape
                    if "batch_shape" in layer_config:
                        layer_config["batch_input_shape"] = layer_config.pop("batch_shape")

                    # ✅ Remove problematic fields
                    for key in ["optional", "dtype_policy", "quantization_config"]:
                        if key in layer_config:
                            del layer_config[key]

                # 🔁 Recurse deeper
                for v in obj.values():
                    fix(v)

            elif isinstance(obj, list):
                for i in obj:
                    fix(i)

        # 👉 APPLY FIX
        fix(config)

        # ✅ SAVE BACK
        f.attrs["model_config"] = json.dumps(config).encode("utf-8")

    print("✅ Fixed successfully!")

# ============================
# 🔥 FIX BOTH MODELS
# ============================

fix_model("letter_model.h5")
fix_model("word_model.h5")