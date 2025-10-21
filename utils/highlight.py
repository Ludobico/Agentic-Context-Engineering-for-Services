from typing import Optional, Literal

def highlight_print(
    target: str, 
    color: Optional[Literal['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'none']] = None,
    **kwargs
):
    # Define ANSI color codes
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print('-' * 80)
    
    # Match the color and apply formatting
    if color is None or color == 'none':
        print(str(target), **kwargs)
    elif color == 'red':
        print(RED + str(target) + RESET, **kwargs)
    elif color == 'green':
        print(GREEN + str(target) + RESET, **kwargs)
    elif color == 'blue':
        print(BLUE + str(target) + RESET, **kwargs)
    elif color == 'yellow':
        print(YELLOW + str(target) + RESET, **kwargs)
    elif color == 'magenta':
        print(MAGENTA + str(target) + RESET, **kwargs)
    elif color == 'cyan':
        print(CYAN + str(target) + RESET, **kwargs)
    elif color == 'white':
        print(WHITE + str(target) + RESET, **kwargs)
    else:
        print(f"Unknown color '{color}', printing without color:")
        print(str(target), **kwargs)

    print('-' * 80)