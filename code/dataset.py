import json
from collections import Counter

class trigger_detection_set():
    def __init__(self, path):
        '''

        :param path: data path
        '''
        f = open(path)
        datas = json.load(f)
        self.raw_data = [(item['words'], item['events']) for item in datas]
        self.sentences = [item['words'] for item in datas]
        self.all_tags = self.__get_tags__() # tag=1: trigger tag=0: Non_trgger
        self.data = [(sentence, tags) for sentence, tags in zip(self.sentences, self.all_tags)]
        self.tag_set, self.tag_counter = self.__get_tagset__()

    def __get_tags__(self):
        all_tags = []
        for item in self.raw_data:
            events = item[1]
            tag = [0] * len(item[0])
            for event in events:
                trigger = event['trigger']
                start, end = int(trigger['start']), int(trigger['end'])
                for i in range(start, end):
                    tag[i] = 1
            if len(tag) != len(item[0]):
                assert False
            all_tags.append(tag)
        return all_tags

    def __get_tagset__(self):
        tag_set = set()
        counter = Counter()
        for tags in self.all_tags:
            for tag in tags:
                tag_set.add(tag)
                counter.update([tag])
        return  tag_set, counter

class trigger_classification_set():
    def __init__(self, path):
        f = open(path)
        datas = json.load(f)
        self.raw_data = [(item['words'], item['events']) for item in datas]
        self.sentences = [item['words'] for item in datas]
        self.trigger_pos = self.__get_trigger__()
        self.all_tags = self.__get_tags__()
        self.data = [(sentence, pos, tags) for sentence, pos, tags in zip(self.sentences, self.trigger_pos, self.all_tags)]
        self.tag_set, self.tag_counter = self.__get_tagset__()

    def __get_trigger__(self):
        all_triggers = []
        for item in self.raw_data:
            events = item[1]
            tag = [0] * len(item[0])
            for event in events:
                trigger = event['trigger']
                start, end = int(trigger['start']), int(trigger['end'])
                for i in range(start, end):
                    tag[i] = 1
            if len(tag) != len(item[0]):
                assert False
            all_triggers.append(tag)
        return all_triggers
    def __get_tags__(self):
        all_tags = []
        for item in self.raw_data:
            events = item[1]
            tag = ['None'] * len(item[0])
            for event in events:
                trigger = event['trigger']
                start, end = int(trigger['start']), int(trigger['end'])
                trigger_type = trigger['trigger-type']
                for i in range(start, end):
                    tag[i] = trigger_type
            if len(tag) != len(item[0]):
                assert False
            all_tags.append(tag)
        return all_tags

    def __get_tagset__(self):
        tag_set = set()
        counter = Counter()
        for tags in self.all_tags:
            for tag in tags:
                tag_set.add(tag)
                counter.update([tag])
        return  tag_set, counter

class argument_classification_set():
    def __init__(self, path):
        f = open(path)
        datas = json.load(f)
        self.raw_data = [(item['words'], item['events']) for item in datas]
        self.sentences = [item['words'] for item in datas]
        self.all_events = self.__get_all_events__()
        self.all_arguments = self.__get_arguments__()
        self.data = [(sentence, event, arguments) for sentence, event, arguments in zip(self.sentences, self.all_events, self.all_arguments)]
        self.trigger_patterns = self.__get_patterns__()

    def __get_all_events__(self):
        all_events = []
        for item in self.raw_data:
            events = item[1]
            events_lst = []
            for event in events:
                trigger_info = tuple((event['trigger']['trigger-type'], event['trigger']['start']))
                arg_dict = dict()
                for arg in event['arguments']:
                    arg_dict[arg['role']] = tuple((arg['start'], arg['end']))
                events_lst.append(tuple((trigger_info, arg_dict)))
            all_events.append(events_lst)
        return all_events

    def __get_arguments__(self):
        all_candidates = []
        for item in self.raw_data:
            events = item[1]
            candidates = []
            for event in events:
                arguments = event['arguments']
                for argument in arguments:
                    candidates.append(tuple((argument['start'], argument['end'])))
            all_candidates.append(candidates)
        return all_candidates

    def __get_patterns__(self):
        pattern_dict = dict()
        for item in self.raw_data:
            events = item[1]
            for event in events:
                trigger_type = event['trigger']['trigger-type']
                if trigger_type not in pattern_dict.keys():
                    pattern_dict[trigger_type] = set()
                arguments = event['arguments']
                for argument in arguments:
                    pattern_dict[trigger_type].add(argument['role'])
        for key in pattern_dict.keys():
            pattern_dict[key] = list(pattern_dict[key])
        return pattern_dict

if __name__ == '__main__':
    path = './data/valid.json'
    dataset = argument_classification_set(path)
    print(dataset.all_arguments[0])