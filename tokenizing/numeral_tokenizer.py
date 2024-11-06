numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder = {f'a{i}': i for i in range(num_nodes)}
        self.encoder.update({f'b{i}': num_nodes + i for i in range(num_nodes)})
        self.encoder['-'] = 2*num_nodes
        self.encoder['='] = 2*num_nodes + 1
        self.encoder['>'] = 2*num_nodes + 2

        self.decoder = {i: f'a{i}' for i in range(num_nodes)}
        self.decoder.update({num_nodes + i: f'b{i}' for i in range(num_nodes)})
        self.decoder[2*num_nodes] = '-'
        self.decoder[2*num_nodes+1] = '='
        self.decoder[2*num_nodes+2] = '>'

    def encode(self, x):
        out = []
        arr = x.split('-')
        out.append(self.encoder[arr[0][0]])
        out.append(self.encoder[arr[0][1:]])
        out.append(self.encoder['-'])
        out.append(self.encoder[arr[1][0:-1]])
        out.append(self.encoder['>'])

        return out

    def decode(self, x):
        return [self.decoder[i] for i in x]
