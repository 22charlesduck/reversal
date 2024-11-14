import string
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder = {f'a{i//2}' if i % 2 == 0 else f'b{i//2}': i for i in range(2 * num_nodes)}
        self.encoder['-'] = 2*num_nodes
        self.encoder['='] = 2*num_nodes + 1
        self.encoder['>'] = 2*num_nodes + 2
        self.encoder['$'] = 2*num_nodes + 3
        self.encoder['@'] = 2*num_nodes + 4
        self.encoder['f'] = 2*num_nodes + 5
        current_index = 2*num_nodes + 6
        for char in string.ascii_uppercase:
            self.encoder[char] = current_index
            current_index += 1

        self.decoder = {v: k for k, v in self.encoder.items()}
        # self.decoder = {2*i: f'a{i}' for i in range(num_nodes)}
        # self.decoder.update({2*i+1: f'b{i}' for i in range(num_nodes)})
        # self.decoder[2*num_nodes] = '-'
        # self.decoder[2*num_nodes+1] = '='
        # self.decoder[2*num_nodes+2] = '>'
        # self.decoder[2*num_nodes+3] = '$'

    def encode(self, x):
        out = []
        num = ''
        for c in x:
            if c == 'a' or c == 'b':
                num = c
            elif c in numbers:
                num += c
            else:
                if num != '':
                    out.append(self.encoder[num])
                    num = ''
                out.append(self.encoder[c])
        if num != '':
            out.append(self.encoder[num])
        return out

    def decode(self, x):
        return [self.decoder[i] for i in x]

# t = NumeralTokenizer(100)
# print(t.encode('IQCMC=a0-b0>'))