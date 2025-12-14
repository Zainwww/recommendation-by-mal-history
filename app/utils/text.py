
def truncate(x, max_len=20):
    return x[:max_len] + '...' if isinstance(x, str) and len(x) > max_len else x