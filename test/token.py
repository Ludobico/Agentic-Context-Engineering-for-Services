from config.getenv import GetEnv

env = GetEnv()
hf_token = env.get_huggingface_token

print(hf_token)