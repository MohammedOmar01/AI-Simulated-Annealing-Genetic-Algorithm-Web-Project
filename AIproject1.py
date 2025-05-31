from flask import Flask, render_template_string, request, jsonify         # For our interface we will use Flask AND using HTML in our python code 
import io, base64, random, math, copy                                     # also to send our plot to the web page by using a library io and base64
import matplotlib
matplotlib.use("Agg")                                                     # anti grain geometry for backend that render graphics that we will deal later
import matplotlib.pyplot as plt
###########################################################################################################                                                                           
class Package:                
    def __init__(self, x, y, weight, prio, pid):           # the packege class contain the tow points x and y in the kilometers and the priorty,weight and id for each pakege
        self.x = x                                         # we need an id for each pakege just only for clasify each pakege later one if it can exist in the car capacity
        self.y = y
        self.weight = weight
        self.priority = prio
        self.id = pid

    def pos(self):                                          # call it to return the position of the currant pakege and we will use it later on for distance calclating and ploting and route building 
        return (self.x, self.y)

def dist(a, b):                                             # function for calculates the euclidean distance between x and y and using math hypot for the square root
    
    return math.hypot(a[0] - b[0], a[1] - b[1])

def empty_vehicle(idx, cap):                                # function for creating an car with it all parameter so a vehicle number and an max capacity,and the currant capacity in used      
    return {                                                # and also the pakeges assigned to this vehicle and also the planned order in array form
'id': idx, 'capacity': cap, 'used': 0,'packages': [],'route': [] }
#########################route distance and priority calculation##################

def nn_route(pkgs, a =1 , b= 5 ):                            # how can we balanced between our proirty and distance , ok what we did is the nearest neighbor function
    if not pkgs:   
                                                            # that it will be called in the inital state before the cars goes to the alogthims and thin inside each alogthim
        return []                                           # so we will creat a array or list then we will see does the score of this pakege is less or more tha the next pakege in another word , did i go to this pakege that have this number of prairty or to this pakege that have a less distance and low praiorty                             
    cur = (0, 0) 

    left =pkgs.copy()                                       # so if there is no pakeges return empty route then give me the point 0,0 in x,y and assign the bakeges we didnt visted yet and let r be the final route in list              
                                                            
    r = []                                                  # while there is an pakege not visted yet so give me the best next route from the array of the not visted pakege , so we will chosse the pakege that gives us the minimum score 
    while left:                                             
        best = min(left, key=lambda p: a * dist(cur, p.pos()) + b * p.priority)

        r.append(best)
                                              # so chose from the unvisted list for every pakege calculate it score by multibly the distance from the currant to the pakege (by euclidean) by a constant , this constant will determine te important of the distance , and adding and the proirty of that pakege (and also multibly it by a constant)
        left.remove(best)                                   # so if we have the smallest score then its the best next pakege by balancing the praiorty and the distance
                                                            # then add this pakege to the dilivary plan and remove it from the unvisted pakeges and our location must move to this pakege 
        cur =best.pos()
    return r
###########################################################################################################                                                                           
def rebuild(vehicles, a= 1 , b =5 ): 
                                                    # important function for rebuld our route when we will swap or move and make any change in any car pakeges list
    for v in vehicles :

        v['route']=  nn_route(v['packages'],  a , b )
###########################################################################################################                                                                           
def capacity_ok(vehicles):                                  # Function to see if the capacity is good or not in each car and return true if the car is good wighted

    return all( v[ 'used'] <= v['capacity' ] for v in vehicles )
###########################################################################################################
def score( vehicles ):
                                          # score function to calcutle the total cost of delevary for each car by its distance and the praiorty
    total= 0

    depot= (0, 0)

    for v in vehicles:

        r = v['route']

        if not r:

            continue

        total += dist(depot, r[0].pos()) * r[0].priority
        
        for p, q in zip(r, r[1:]):

            total += dist(p.pos(), q.pos()) * q.priority

    return total

###########################################################################################################
def true_cost(vehicles): 
                                     # ourr true cost will calclate the true final path from the shop to back to the shop
    total = 0

    depot = (0, 0)
    for v in vehicles:

        r = v['route']

        if not r:

            continue
       
        total += dist(depot, r[0].pos())
       
        for p, q in zip(r, r[1:]):

            total += dist(p.pos(), q.pos())
        
        total += dist(r[-1].pos(), depot)

    return total
###########################################################################################################
def sa_initial(caps, pkgs, a, b):                       # start our sa alsogthim fut initialy we need to assign a random pakege to a random car  
                                                          # for getting a little randomnes to the alogthim , and give us an near expected value path
    vehs = [empty_vehicle(i + 1, c) for i, c in enumerate(caps)]
                                                         # so after we assing accorfing to random , and (capicity for each car) we send each list car pakeges to the score function that we discripe it before
                                                          # and the score function will give us every car with the best path of each pakeges ( and this is the start point)
    random.shuffle(pkgs)
    undelivered = []

    for p in pkgs:
        assigned = False

        for v in random.sample(vehs, len(vehs)):

            if v['used'] + p.weight <= v['capacity']:

                v['packages'].append(p)

                v['used'] += p.weight                  # if we add or shift or swap or doing any think to each car , dirictly we see the capicty of it if it good or bigger , if bigger , dont do this action
                assigned = True

                break
            if not assigned:
              
              undelivered.append(p)


    rebuild(vehs, a, b)

    return vehs, undelivered
###########################################################################################################
def sa_neighbor(cur, a, b, tries=10):                   # the secand function for make a neighbor solution by randomly trying small change
                                                        # we have 4 think we will do move pakege between cars 
    for _ in range(tries):                              # swap tow pakeges between tow random car 
                                                        # shuffle packages inside one vehicle
        nb = copy.deepcopy(cur)                         # reverse a slice of the packages inside one vehicle

        op = random.choice(['move', 'swap', 'shuffle', 'reverse'])

        if op == 'move':

            v_from = random.choice(nb)

            if not v_from['packages']:
                continue
            p = random.choice(v_from['packages'])

            v_from['packages'].remove(p)

            v_from['used'] -= p.weight

            for v_to in random.sample(nb, len(nb)):

                if v_to['used'] + p.weight <= v_to['capacity']:

                    v_to['packages'].append(p)

                    v_to['used'] += p.weight

                    break
            else:
                continue
        elif op == 'swap':

            v1, v2 = random.sample(nb, 2)

            if not v1['packages'] or not v2['packages']:

                continue
            p1 = random.choice(v1['packages'])

            p2 = random.choice(v2['packages'])

            if (v1['used'] - p1.weight + p2.weight <= v1['capacity'] and
                
                    v2['used'] - p2.weight + p1.weight <= v2['capacity']):
                
                v1['packages'].remove(p1); v2['packages'].remove(p2)

                v1['packages'].append(p2); v2['packages'].append(p1)

                v1['used'] += p2.weight - p1.weight

                v2['used'] += p1.weight - p2.weight
            else:
                continue
        elif op == 'shuffle':

            random.shuffle(random.choice(nb)['packages'])

        else:  

            big = [v for v in nb if len(v['packages']) >= 4]

            if not big:

                continue
            v = random.choice(big)

            i, j = sorted(random.sample(range(len(v['packages'])), 2))

            v['packages'][i:j] = reversed(v['packages'][i:j])

        rebuild(nb, a, b)

        if capacity_ok(nb):

            return nb
    return cur  
###########################################################################################################
def sa(caps, pkgs, a=1.0, b=5.0, T0=1000, cool=0.95, inner=100, T_stop=1):
                                                     # main simulated annealing algorithm
    cur, undelivered = sa_initial(caps, pkgs, a, b)          # start with an initial random solution our sa_initial 

    best = copy.deepcopy(cur)                   # copy the best path for each car after in sa_intial function will call the score function 
                                                # way we need it , since if the the next temp itration give a bad score but in later on temp , it will have a copy of the path and pakeges to not let the new low score
    c_cur = score(cur)

    c_best = c_cur
    T = T0
    while T > T_stop:                           # T_stop is 1 , for every T > than 1 then we will start out ittration 
        for _ in range(inner):                  # try the inner times to move to a neighbor solution (sa_neighbr function)
                                                # stop when temperature becomes lower than T_stop
            nb = sa_neighbor(cur, a, b)        

            if nb is cur:
                continue
                                                # always accept better solutions (according to the score function)
            c_nb = score(nb)
            if c_nb < c_cur or random.random() < math.exp((c_cur - c_nb) / T):
                                                # accept a worse solution with some probability depending on T (when T is low , these worse case will gone)
                cur, c_cur = nb, c_nb        
                if c_cur < c_best:

                    best, c_best = copy.deepcopy(cur), c_cur
        T *= cool                             # always lower the temperature by a cooling factor the cool 0.95
    return best, c_best, undelivered                      # return the best found solution and its internal cost
###########################################################################################################
class GA:                                      # main genetic algorithm
    def __init__(self, caps, pkgs, pop_size=80, mutation_rate=0.05, generations=500,a=1.0, b=5.0):
        self.caps, self.pkgs = caps, pkgs              # caps is the list of vehicle capacities and pkgs list of Package objects
                                                       # pop_size is number of chromosomes in the population and  mutation_rate is chance of mutation per child
        self.pop_size, self.mut = pop_size, mutation_rate # generations is the number of times the population evolves

        self.generations, self.a, self.b = generations, a, b
###########################################################################################################    
    def _random_chrom(self):                    #create a random chromosome (random sequence of packages)
                                               #this is one guess for how to assign packages
        chrom = self.pkgs[:]

        random.shuffle(chrom)
        return chrom
###########################################################################################################
    def de(self, chrom):                       # convert a chromosome (package sequence) into a list of vehicles and ry to assign packages greedily to the first vehicle that can carry them
        vehs = [empty_vehicle(i + 1, c) for i, c in enumerate(self.caps)]
        undelivered = []                                      #  if package fits put it in the vehicle according to the capicty , that alwyes will ack for cheack
        for p in chrom:
            
            assigned = False                          
            for v in vehs:  
                      
                if v['used'] + p.weight <= v['capacity']:

                    v['packages'].append(p)

                    v['used'] += p.weight
                    assigned = True

                    break

            if not assigned:

                undelivered.append(p)  # ← added to track undelivered packages

        rebuild(vehs, self.a, self.b)

        return vehs, undelivered  # ← modified to also return undelivered list

###########################################################################################################
    def fitness(self, chrom):               # the fitness is the score function we have as chromosome
                                            # decode it to vehicle assignments and routes then return the total score if all vehicles are within "capacity
        vehs, _ = self.de(chrom)            # but else return a hug number

        return score(vehs) if capacity_ok(vehs) else 1e8

###########################################################################################################
    def tournament(self, pop, k=3):        # select a parent chromosome using tournament selection then pick k random chromosomes

        return min(random.sample(pop, k), key=self.fitness)  # return the one with the best lowest fitness score
###########################################################################################################
    def crossover(self, p1, p2):          # do a  crossover bwtween parent 1 and perant 2 

        n = len(p1)                       # copy a slice or a part from p1 into the child fill the rest from p2, keeping the order and deteccr duplicte
        a, b = sorted(random.sample(range(n), 2))

        child = [None] * n                

        child[a:b] = p1[a:b]

        fill = [g for g in p2 if g not in child]

        idx = 0
        for i in range(n):

            if child[i] is None:

                child[i] = fill[idx]; idx += 1

        return child                  # result is a new child chromosome package order
###########################################################################################################
    def mutate(self, chrom):             # mutate a chromosome with a small chance randomly swap two packages in the chromosome

        if random.random() < self.mut:   # this introduces randomness to escape local optima

            i, j = random.sample(range(len(chrom)), 2)

            chrom[i], chrom[j] = chrom[j], chrom[i]
###########################################################################################################
    def run(self):                        # main GA algorithm 
                                        # generate initial random population
        pop = [self._random_chrom() for _ in range(self.pop_size)]
                                        # and for each generation create new population using selection and crossoverand mutation.
        for _ in range(self.generations):
                                        # after all generations return the best solution found
            new_pop = []

            while len(new_pop) < self.pop_size:

                p1 = self.tournament(pop)

                p2 = self.tournament(pop)

                child = self.crossover(p1, p2)

                self.mutate(child)

                new_pop.append(child)

            pop = new_pop

        best = min(pop, key=self.fitness)

        vehs, undelivered = self.de(best)   #  added to capture undelivered
        
        return vehs, self.fitness(best), undelivered  # modified to return all

###########################################################################################################
PAGE = """                          
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Delivery Optimiser</title>
<style>
body{font-family:system-ui,sans-serif;background:#f7fafc;margin:0;display:flex;justify-content:center;padding:2rem}
.card{background:#fff;border-radius:1rem;box-shadow:0 4px 16px rgba(0,0,0,.08);padding:2rem;max-width:960px;width:100%}
h1{text-align:center;margin-top:0}
input[type=number]{width:100%;padding:.45rem .5rem;border:1px solid #cbd5e0;border-radius:.35rem;font-size:.9rem}
input.error{border-color:#e53e3e;background:#fff5f5}
label{font-size:.85rem;color:#4a5568}
fieldset{border:1px solid #e2e8f0;border-radius:.6rem;padding:1rem;margin-bottom:.7rem;background:#f9fbfd}
legend{font-weight:600;color:#2d3748;padding:0 .5rem}
.package-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:.6rem}
.btn-row{display:flex;justify-content:space-between;margin-top:1.2rem}
button{flex:1 1 45%;padding:.8rem;border:0;border-radius:.5rem;font-size:1rem;font-weight:600;cursor:pointer;transition:background .2s}
button#saBtn{background:#2b6cb0;color:#fff} button#gaBtn{background:#718096;color:#fff}
button:hover{opacity:.92}
hr{margin:1.5rem 0;border:none;border-top:1px solid #e2e8f0}
#result img{max-width:100%;margin-top:1rem;border:1px solid #cbd5e0;border-radius:.4rem}
</style>
</head>
<body>
<div class='card'>
 <h1>Delivery Optimiser</h1>
 <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:1rem'>
   <div><label>Number of vehicles</label><input id='numV' type='number' min='1' value='3'></div>
   <div><label>Number of packages</label><input id='numP' type='number' min='1' value='10'></div>
 </div>
 <button style='margin-top:1rem' onclick='genFields()'>Create Fields</button>
 <hr>
 <form id='dataForm'><div id='vehicles'></div><div id='packages'></div></form>
 <div class='btn-row'>
   <button id='saBtn' onclick='runAlgo("sa")'>Run SA</button>
   <button id='gaBtn' onclick='runAlgo("ga")'>Run GA</button>
 </div>
 <div id='result'></div>
</div>
<script>
function genFields(){
  const nV = +document.getElementById('numV').value || 0;
  const nP = +document.getElementById('numP').value || 0;
  const vDiv = document.getElementById('vehicles');
  const pDiv = document.getElementById('packages');
  vDiv.innerHTML = ''; pDiv.innerHTML = '';
  for(let i = 1; i <= nV; i++){
    vDiv.insertAdjacentHTML('beforeend',
      `<div style='margin-bottom:.8rem'>
         <label style='font-weight:600'>Vehicle ${i} capacity (kg)</label>
         <input type='number' name='cap_${i}' required>
       </div>`);
  }
  for(let j = 1; j <= nP; j++){
    pDiv.insertAdjacentHTML('beforeend',
      `<fieldset><legend>Package ${j}</legend>
         <div class='package-row'>
           <div><label>x</label><input type='number' step='any' name='x_${j}' required></div>
           <div><label>y</label><input type='number' step='any' name='y_${j}' required></div>
           <div><label>Weight</label><input type='number' name='w_${j}' required></div>
           <div><label>Priority&nbsp;(1-5)</label><input type='number' name='prio_${j}' min='1' max='5' required></div>
         </div>
       </fieldset>`);
  }
}

function runAlgo(kind){
  const form = document.getElementById('dataForm');
  let ok = true;
  form.querySelectorAll('input').forEach(i => {
    if(!i.value){ i.classList.add('error'); ok = false; }
    else        { i.classList.remove('error'); }
  });
if (!ok) {
  alert('Fill highlighted fields');
  return;
}
const fd = new FormData(form);
fd.append('numVehicles', document.getElementById('numV').value);
fd.append('numPackages', document.getElementById('numP').value);

fetch('/run_' + kind, { method: 'POST', body: fd })
  .then(r => r.json())
  .then(d => {
    document.getElementById('result').innerHTML =
      `<h3>Final Delivery Cost = ${d.cost.toFixed(2)} km</h3>` +
      // Removed: `<p style="color:gray;font-size:.9em"> Optimizer Score = ${d.internal_score.toFixed(2)}</p>` +
      d.summary +
      `<img src='data:image/png;base64,${d.plot}'>`;
  });

}

window.onload = genFields;
</script>
</body>
</html>
"""
###########################################################################################################
app = Flask("mohammadomar")


def parse_form(f):

    nV = int(f['numVehicles']); nP = int(f['numPackages'])

    caps = [int(f[f'cap_{i}']) for i in range(1, nV + 1)]

    pkgs = [Package(float(f[f'x_{j}']), float(f[f'y_{j}']), float(f[f'w_{j}']), float(f[f'prio_{j}']), j)
            
            for j in range(1, nP + 1)]
    
    return caps, pkgs

def make_plot(best, title="Optimised Routes", undelivered=None):
    fig, ax = plt.subplots(figsize=(7, 4))  


    ax.plot(0, 0, 'ks', label='Shop')       

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    
    for v in best:


        if not v['route']:

            continue
        route = [(0, 0)] + [p.pos() for p in v['route']] + [(0, 0)]

        xs, ys = zip(*route)

        ax.plot(xs, ys, 'o-', color=colors[(v['id'] - 1) % len(colors)],
                
                label=f'Vehicle {v["id"]}')

                                                        # for display undeliveredd package as text if there is 

    if undelivered:
        text = "Undelivered Packages:\n" + "\n".join(

            [f"ID {p.id} ({p.weight}kg)" for p in undelivered]
        )
        ax.text(-25, 95, text, fontsize=8, color="red", ha='left', va='top')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_summary(best, undelivered=None):

    s = "<ul style='list-style:none;padding:0'>"

    delivered_ids = set()

    
    for v in best:

        if not v['route']:

            s += f"<li>Vehicle {v['id']}: idle (capacity {v['capacity']} kg)</li>"

        else:

            ids = [p.id for p in v['route']]

            delivered_ids.update(ids)


            s += (f"<li>Vehicle {v['id']}: used {v['used']:.3f}/{v['capacity']} kg → {ids}</li>")

                                          
    if undelivered:

        seen_ids = set()

        truly_undelivered = []

        for p in undelivered:
            if p.id not in delivered_ids and p.id not in seen_ids:

                truly_undelivered.append(p)
                seen_ids.add(p.id)

        if truly_undelivered:
            s += "<li><b>Undelivered Packages:</b> "
            s += ", ".join([f"Pakage {p.id} ({p.weight:.1f}kg)" for p in truly_undelivered])
            s += "</li>"

    s += "</ul>"
    return s


@app.route('/')

def home():

    return render_template_string(PAGE)

@app.route('/run_sa', methods=['POST'])

def run_sa_route():

    caps, pkgs = parse_form(request.form)

    best, internal_cost, undelivered = sa(caps, pkgs)


    final_cost = true_cost(best)

    return jsonify({

        'cost': final_cost,'internal_score': internal_cost,'summary': make_summary(best, undelivered), 'plot': make_plot(best, "Simulated Annealing Plot")   })

@app.route('/run_ga', methods=['POST'])

@app.route('/run_ga', methods=['POST'])
def run_ga_route():
    caps, pkgs = parse_form(request.form)

    ga = GA(caps, pkgs, pop_size=80, mutation_rate=0.05, generations=500)

    best, internal_cost, undelivered = ga.run()

    final_cost = true_cost(best)

    return jsonify({'cost': final_cost, 'internal_score': internal_cost, 'summary': make_summary(best, undelivered), 'plot': make_plot(best, "Genetic Algorithm Plot")
    })

if __name__ == '__main__':


    print("  Delivery Optimizer starting...")

    print(" Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=False)
