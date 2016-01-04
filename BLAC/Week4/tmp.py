

def ipv4_to_hex(ipv4):
    ipv4_hex = str()
    individual_bytes = ipv4.split('.')

    for byte in individual_bytes:
        ipv4_hex += str(hex(int(byte))) + '.'

    print ipv4_hex.split('.')

    for byte in ipv4_hex.split('.'):
        ipv4_byte[1-3]

    return ipv4_hex[:-1]


def main():
    print ipv4_to_hex('192.168.1.102')


if __name__ == '__main__':
    main()
