import sys
import pprint

class KNMIParser(object):
    def __init__(self, filename):
        self.filename = filename
        self.n_columns = None
    
        
    def parse(self):
        with open(self.filename, 'r') as file:
            stations, metadata, columns = self.parse_header(file)
            data = self.parse_body(file)
        
        return stations, metadata, columns, data

    def parse_header(self, file):
        self.skip(file)
        stations = self.parse_stations(file)
        metadata = self.parse_metadata(file)
        columns = self.parse_column_header(file)

        return stations, metadata, columns

    def skip(self, file):
        for line in file:
            if line.startswith('# STN'):
                break

    def parse_stations(self, file):
        out = []

        for line in file:
            split_line = line[2:].split()
            if len(split_line) < 5:
                break
            
            station = int(split_line[0][:-1])
            longitude = float(split_line[1])
            latitude = float(split_line[2])
            altitude = float(split_line[3])
            name = ' '.join(split_line[4:])

            out.append((station, longitude, latitude, altitude, name))
        return out

    def parse_metadata(self, file):
        out = {}

        for line in file:
            split_line = line[2:].split()
            if len(split_line) == 0:
                break

            key = split_line[0]
            value = ' '.join(split_line[2:])

            out[key] = value
        return out

    def parse_column_header(self, file):
        for line in file:
            columns = [x.strip() for i, x in enumerate(line[2:].split(','))]
            break
        self.n_columns = len(columns)

        return columns

    def parse_body(self, file):
        rows = []
        n_rows = 0

        for line in file:
            line = line.strip()
            if line and line[0] != '#':
                split_line = line.split(',')
                numbers = []

                for chunk in split_line:
                    chunk = chunk.strip()
                    if chunk == '':
                        numbers.append(None)
                    else:
                        numbers.append(int(chunk))


                assert len(numbers) == self.n_columns
                rows.append(tuple(numbers))
                n_rows += 1
                if n_rows % 10000 == 0:
                    print 'x'

        print
        return rows

class DataSet(object):
    def __init__(self, stations, metadata, columns, data):
        self.stations = stations
        self.metadata = metadata
        self.columns = columns
        self.data = data
    


if __name__ == '__main__':
    # inspection of the file shows all numbers to be integers :)
    parser = KNMIParser(sys.argv[1])
    ds = DataSet(*parser.parse())
    idx = ds.columns.index('RH')  # gives the index of a certain variable in each row

    print ds.data[0][idx]  # access the RH variable in the first row of data
