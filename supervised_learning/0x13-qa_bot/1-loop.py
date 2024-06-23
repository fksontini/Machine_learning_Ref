#!/usr/bin/env python3
"""Take input from the user with the prompt 
Q: and prints A: as a response"""
exit = ['exit', 'quit', 'goodbye', 'bye']
while True:
    print('Q:', end='')
    question = input()
    if question.lower() in exit:
        print('A: Goodbye')
        break
    print('A:',)
