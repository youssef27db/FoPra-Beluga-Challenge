# trainer/training.py
from agent.ppo.ppo_agent import PPOAgent
from trainer.ppo_trainer import PPOTrainer
from env.env import Env

# Mappings von kommandozeilen zu internen bezeichnungen

AGENTS = {
    "ppo": PPOAgent,
    # "dqn": DQNAgent, ...
}

TRAINERS = {
    "ppo": PPOTrainer,
    # "dqn": DQNTrainer, ...
}


AGENT_KWARGS = {
    "ppo": lambda args: {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "learning_rate": args.learning_rate,
        "clip_rate": args.clip_rate
    },
    "dqn": lambda args: {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "learning_rate": args.learning_rate,
        "epsilon_start": args.epsilon_start,
        "epsilon_decay": args.epsilon_decay
    },
}




def run(args):

    # Kommandozeilenargumente verarbeiten mit warnung falls nicht vorhanden oder falsch geschrieben
    model_key = args.model.lower()
    if model_key not in AGENTS or model_key not in TRAINERS:
        raise ValueError(f"Unbekannter Model-Typ: {args.model}")

    # Initialisiere die Umgebung und deklariere Agenten/Trainer basierend auf dem Modelltyp
    env = Env()
    agent_class = AGENTS[model_key]
    trainer_class = TRAINERS[model_key]

    # initialisiere Agenten und Trainer

    agent_kwargs = AGENT_KWARGS[model_key](args)

    
    agent = agent_class(**agent_kwargs)
    trainer = trainer_class(env, agent, **agent_kwargs)

    # Starte das Training
    trainer.train(episodes=args.episodes)


