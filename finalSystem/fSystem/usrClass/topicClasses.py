class hasGoods:
    def __init__(self,place,occupy,goods):
        self.place,self.occupy,self.goods = place,occupy,goods
    def getTopicSentence(self):
        if len(self.occupy)>0:
            return '{self.place}{self.occupy}{self.goods}。'.format(self=self)
        else:
            return '{self.place}有{self.goods}。'.format(self=self)

class Damage:
    def __init__(self,public,damage):
        self.public,self.damage = public,damage
    def getTopicSentence(self):
        return '{self.public}{self.damage}。'.format(self=self)
class Activity:
    def __init__(self,org,activity):
        self.org,self.activity = org,activity
    def getTopicSentence(self):
        return '{self.org}组织开展{self.activity}。'.format(self=self)
class Consult:
    def __init__(self,con,cer):
        self.con,self.cer = con,cer
    def getTopicSentence(self):
        return '{self.con}{self.cer}。'.format(self=self)
class Visit:
    def __init__(self,v,org):
        self.v,self.org = v,org
    def getTopicSentence(self):
        return'{self.v}:{self.org}。'.format(self=self).rstrip(':')
