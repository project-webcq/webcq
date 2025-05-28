



# MARL System Runner

This project provides a framework for running Multi-Agent Reinforcement Learning (MARL) experiments with customizable profiles and session tracking.

## 📦 Installing Dependencies

Make sure you have **Python 3.8+** installed. Install all required Python packages using:

```bash
pip install -r requirements.txt
```

## 📌 Example Use Case

You can run the main experiment script with a specific configuration using the following command:

```bash
python main.py --profile=github-marl-3h-qtran-5agent --session=1
```

This command will run WebCQ (qtran) on github for 3h.

Also, you can run our implementation of MARG (DQL setting) with the following command：

```bash
python main.py --profile=github-marl-3h-marg-dql-5agent --session=1
```

| Argument    | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `--profile` | Specifies the experiment configuration file (located in `./settings.yaml`). |
| `--session` | Custom session name to separate logs and results.            |

## 🧠 Notes

Ensure the profile specified by `--profile` exists in the `settings.yaml` .

Logs and results are stored under `default_output_path`, defined in `./settings.yaml`.

Run multiple experiments by changing either the profile or the session name.
