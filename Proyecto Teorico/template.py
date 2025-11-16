################################################
# Template for ScotoMa model implementation
################################################

# Import necessary libraries and modules
import numpy as np
import scipy as scp
import pandas as pd
import numpy as np
from casas_ibarra_scotoMa import casas_ibarra 
from smmcmc import smmcmc


################################# Default functions for every scoto analysis ##########################################\

#function to compute lam_1 from mH and v
def lam_1fixed(parameters):
    others_block=parameters["OTHER"]
    lam_1=(others_block["mH"]**2)/(others_block["v"]**2)
    return lam_1

################################# Default constrains for every scoto analysis ##########################################\

def stability_constraints(blocks):
    inputblock = blocks["MINPAR"]
    lambda_1 = inputblock["lambda1Input"]
    lambda_2 = inputblock["lambda2Input"]
    lambda_3 = inputblock["lambda3Input"]
    lambda_4 = inputblock["lambda4Input"]
    lambda_5 = inputblock["lambda5Input"]

    # Condición 1: λ1 ≥ 0 y λ2 ≥ 0
    # Garantiza que los potenciales en las direcciones puras de los campos sean estables.
    cond1 = (lambda_1 >= 0) and (lambda_2 >= 0)

    # Condición 2: λ3 ≥ -2 * sqrt(λ1 * λ2)
    # Evita que se formen direcciones inestables cuando ambos campos adquieren valores grandes.
    cond2 = (lambda_3 >= -2 * np.sqrt(lambda_1 * lambda_2))

    # Condición 3: (λ3 + λ4 ± λ5)/2 ≥ - sqrt(λ1 * λ2)
    # Controla las combinaciones mixtas entre los campos,
    # asegurando que el potencial sea acotado por debajo en direcciones arbitrarias.
    cond3 = ((0.5 * (lambda_3 + lambda_4 + lambda_5) >= -np.sqrt(lambda_1 * lambda_2)) and
             (0.5 * (lambda_3 + lambda_4 - lambda_5) >= -np.sqrt(lambda_1 * lambda_2)))

    return (cond1 and cond2 and cond3)


def unitarity_constraints(blocks):
    inputblock = blocks["MINPAR"]
    lambda_1 = inputblock["lambda1Input"]
    lambda_2 = inputblock["lambda2Input"]
    lambda_3 = inputblock["lambda3Input"]
    lambda_4 = inputblock["lambda4Input"]
    lambda_5 = inputblock["lambda5Input"]

    bound = 8 * np.pi

    # Condición 1: |λ3 ± λ4| ≤ 8π
    # Restricciones sobre combinaciones lineales simples entre λ3 y λ4.
    cond1 = (np.abs(lambda_3 + lambda_4) <= bound) and (np.abs(lambda_3 - lambda_4) <= bound)

    # Condición 2: |λ3 ± λ5| ≤ 8π
    # Restricciones sobre combinaciones lineales simples entre λ3 y λ5.
    cond2 = (np.abs(lambda_3 + lambda_5) <= bound) and (np.abs(lambda_3 - lambda_5) <= bound)

    # Condición 3: |λ3 + 2λ4 ± 3λ5| ≤ 8π
    # Controla combinaciones con mayor peso relativo de λ4 y λ5.
    cond3 = (np.abs(lambda_3 + 2*lambda_4 + 3*lambda_5) <= bound) and (np.abs(lambda_3 + 2*lambda_4 - 3*lambda_5) <= bound)

    # Condición 4: |λ1 + λ2 ± sqrt((λ1 − λ2)^2 + λ4)| ≤ 8π
    # Relaciona λ1 y λ2 con corrección cuadrática proveniente de λ4.
    sqrt1 = np.sqrt((lambda_1 - lambda_2)**2 + lambda_4)
    cond4 = (np.abs(lambda_1 + lambda_2 + sqrt1) <= bound) and (np.abs(lambda_1 + lambda_2 - sqrt1) <= bound)

    # Condición 5: |3λ1 + 3λ2 ± sqrt(9(λ1 − λ2)^2 + (2λ3 + λ4)^2)| ≤ 8π
    # Extiende la condición anterior pero con factores de 3 y dependencias adicionales de λ3 y λ4.
    sqrt2 = np.sqrt(9*(lambda_1 - lambda_2)**2 + (2*lambda_3 + lambda_4)**2)
    cond5 = (np.abs(3*lambda_1 + 3*lambda_2 + sqrt2) <= bound) and (np.abs(3*lambda_1 + 3*lambda_2 - sqrt2) <= bound)

    # Condición 6: |λ1 + λ2 ± sqrt((λ1 − λ2)^2 + λ5)| ≤ 8π
    # Similar a la condición 4, pero con λ5 en lugar de λ4.
    sqrt3 = np.sqrt((lambda_1 - lambda_2)**2 + lambda_5)
    cond6 = (np.abs(lambda_1 + lambda_2 + sqrt3) <= bound) and (np.abs(lambda_1 + lambda_2 - sqrt3) <= bound)

    return (cond1 and cond2 and cond3 and cond4 and cond5 and cond6)

def perturbativity(blocks):
    inputblock=blocks["MINPAR"]
    lambda_1=inputblock["lambda1Input"]
    lambda_2=inputblock["lambda2Input"]
    lambda_3=inputblock["lambda3Input"]
    lambda_4=inputblock["lambda4Input"]
    lambda_5=inputblock["lambda5Input"]
    return((abs(lambda_2)<4*np.pi) and (abs(lambda_3)<4*np.pi) and (abs(lambda_4)<4*np.pi) and (abs(lambda_5)<4*np.pi))

def small_lam5(blocks):
    inputblock=blocks["MINPAR"]
    lambda_1=inputblock["lambda1Input"]
    lambda_2=inputblock["lambda2Input"]
    lambda_3=inputblock["lambda3Input"]
    lambda_4=inputblock["lambda4Input"]
    lambda_5=inputblock["lambda5Input"]
    #This condition is not in the paper, but it is a good idea to avoid large values of lam5.
    return(lambda_5<lambda_1 and lambda_5<lambda_2 and lambda_5<lambda_3 and lambda_5<lambda_4)

def order_maj_masses(blocks):
    mass_block=blocks["MNIN"]
    m_Chi_1= -1*mass_block["Mn(1,1)"]
    m_Chi_2= -1*mass_block["Mn(2,2)"]
    m_Chi_3= -1*mass_block["Mn(3,3)"]
    return (m_Chi_1<m_Chi_2 and m_Chi_2<m_Chi_3)

def postitive_scalar_mass(blocks):
    minpar=blocks["MINPAR"]
    other=blocks["OTHER"]
    mEt2=minpar["mEt2Input"]
    lambda3=minpar["lambda3Input"]
    lambda4=minpar["lambda4Input"]
    lambda5=minpar["lambda5Input"]
    v=other["v"]
    metR2=mEt2**2 + (lambda3+lambda4+lambda5)*v**2
    metI2=mEt2**2 + (lambda3+lambda4-lambda5)*v**2
    return (metR2>0 and metI2>0)

def valid_yukawa(params):
    ynin = casas_ibarra(params)["YNIN"]
    for i in range(1, 4):
        for j in range(1, 4):
            yukawa = ynin[f"Yn({i},{j})"]
            # Check if the Yukawa coupling is too small or too large
            if not (1e-10 < abs(yukawa) < 2):
                return False
    return True

#Higgs branching ratio. Using current data from PDG

def BR_higgs_inv(model_decay:dict):

    model_decays=model_decay["DECAY 25"]
    br1=float(model_decays["BR(hh -> Chi_1 Chi_1)"])
    br2=float(model_decays["BR(hh -> Chi_2 Chi_2)"])
    br3=float(model_decays["BR(hh -> Chi_3 Chi_3)"])
    br4=float(model_decays["BR(hh -> etI etI)"])
    br5=float(model_decays["BR(hh -> etR etR)"])

    return ( (br1+br2+br3+br4+br5) <0.107 )

def S_T_U_constraints(blocks):
    low_energy_block=blocks["SPhenoLowEnergy"]
    S=float(low_energy_block["S-parameter (1-loop BSM)"])
    T=float(low_energy_block["T-parameter (1-loop BSM)"])
    U=float(low_energy_block["U-parameter (1-loop BSM)"])
    return (S>(-0.01-0.1)and S<(-0.01+0.1) and T>(0.03-0.12) and T<(0.03+0.12)and U>(0.02-0.11) and U<(0.02+0.11))

def LFV_constraints(blocks):
    LFV_block=blocks["FlavorKitLFV"]
    BR_mu_e=LFV_block["BR(mu->e gamma)"]
    BR_tau_e=LFV_block["BR(tau->e gamma)"]
    BR_tau_mu=LFV_block["BR(tau->mu gamma)"]
    BR_mu_3e= LFV_block["BR(mu->3e)"]
    BR_tau_3e= LFV_block["BR(tau->3e)"]
    BR_tau_3mu= LFV_block["BR(tau->3mu)"]
    BR_tau_e_2mu= LFV_block["BR(tau- -> e- mu+ mu-)"]
    BR_tau_mu_2e= LFV_block["BR(tau- -> mu- e+ e-)"]
    BR_tau_mu_e2= LFV_block["BR(tau- -> mu+ e- e-)"]
    #BR_tau_e_pi= LFV_block["BR(tau->e pi)"]
    #BR_tau_e_eta= LFV_block["BR(tau->e eta)"]
    #BR_tau_e_eta_prime= LFV_block["BR(tau->e eta')"]
    #BR_tau_mu_pi= LFV_block["BR(tau->mu pi)"]
    #BR_tau_mu_eta= LFV_block["BR(tau->mu eta)"]
    #BR_tau_mu_eta_prime= LFV_block["BR(tau->mu eta')"]
    CR_mu_e_Ti=LFV_block["CR(mu-e, Ti)"]
    CR_mu_e_Pb=LFV_block["CR(mu-e, Pb)"]
    CR_mu_e_Au=LFV_block["CR(mu-e, Au)"]
    BR_Z_e_mu=LFV_block["BR(Z->e mu)"]
    BR_Z_e_tau=LFV_block["BR(Z->e tau)"]
    BR_Z_mu_tau=LFV_block["BR(Z->mu tau)"]
    return (BR_mu_e<4.2e-13 and BR_tau_e<3.3e-8 and BR_tau_mu<4.2e-8 and BR_mu_3e<1.0e-12 and BR_tau_3e<2.7e-8 and BR_tau_3mu<2.1e-8 and BR_tau_e_2mu<2.7e-8 and BR_tau_mu_2e<1.8e-8 and BR_tau_mu_e2<1.5e-8 and CR_mu_e_Ti<4.3e-12 and CR_mu_e_Pb<4.3e-11 and CR_mu_e_Au<7.0e-13 and BR_Z_e_mu<7.5e-7 and BR_Z_e_tau<5e-6 and BR_Z_mu_tau<6.5e-6
    and BR_Z_e_mu>0.0 and BR_Z_e_tau>0.0 and BR_Z_mu_tau>0.0 and BR_mu_e>0.0 and BR_tau_e>0.0 and BR_tau_mu>0.0 and BR_mu_3e>0.0 and BR_tau_3e>0.0 and BR_tau_3mu>0.0 and BR_tau_e_2mu>0.0 and BR_tau_mu_2e>0.0 and BR_tau_mu_e2>0.0 )

def any_DM_candidate(blocks):
    massblock=blocks["MASS"]
    maj_blocks=blocks["MNIN"]
    m_Chi_1= -1*maj_blocks["Mn(1,1)"]
    m_Chi_2= -1*maj_blocks["Mn(2,2)"]
    m_Chi_3= -1*maj_blocks["Mn(3,3)"]
    m_EtaI=massblock["etI"]
    m_EtaR=massblock["etR"]
    m_Etapm=massblock["etp"]
    if (m_Chi_1<m_EtaI and m_Chi_1<m_EtaR and m_Chi_1<m_Etapm and m_Chi_1<m_Chi_2 and m_Chi_1<m_Chi_3):
        return True
    elif (m_EtaI<m_Chi_1 and m_EtaI<m_EtaR and m_EtaI<m_Etapm and m_EtaI<m_Chi_2 and m_EtaI<m_Chi_3):
        return True
    else:
        return False

def relic_density_constraint(blocks):
    dm_relic=blocks["MICROMEGAS"]["Omega h^2 (Dark matter relic density)"]
    return (dm_relic<(0.12+0.0036) and dm_relic>0.0)

#function and lists to compute DD and ID constraints

XENONnT_x = np.array([5.97102706329433, 6.056990777910455, 6.160566422181327, 6.289002969105092, 6.5088471281204665, 6.649315180341909, 6.912072847559385, 7.146855289006537, 7.542319101684038, 8.087066424421094, 8.716394690763698, 9.370006512226404, 9.900913550722128, 10.84673824252975, 12.097687802766076, 13.502192361831488, 14.813291425564906, 16.184288986599853, 17.682174911494293, 19.53012730021894, 21.49972419743351, 22.970662542958756, 24.61205840374446, 26.769602191914576, 29.22532137596798, 32.409249165849104, 36.080037270419176, 39.10348669193098, 43.25867328241102, 47.763894297413685, 53.70917014508595, 59.0249988759682, 65.64360614159538, 73.62702189370627, 83.18115289013227, 93.54608094837315, 105.99199228989718, 121.92479306630726, 139.87147841454845, 162.4975786888632, 181.52582857500772, 202.21909085340548, 226.4262048439145, 252.6002626785199, 291.3108631068599, 333.3020301259782, 372.05300195829216, 421.79266907745773, 481.42467348270446, 554.7707562117209, 626.9474392737438, 747.7640714442148, 942.0297063780309])
XENONnT_y = np.array([2.5760550137487362e-45, 2.257052231065699e-45, 1.8644568822803518e-45, 1.4928965949933253e-45, 1.105617534806812e-45, 8.323210001311656e-46, 6.0876924262739836e-46, 4.345504331686108e-46, 2.7671708023896064e-46, 1.572951366283984e-46, 8.969554982853953e-47, 5.392749631502421e-47, 3.5220318887703177e-47, 2.198924380469394e-47, 1.2602825785084277e-47, 7.820726293337094e-48, 5.583459697094887e-48, 4.2515168349127356e-48, 3.384706823829771e-48, 2.5917493016275825e-48, 2.200899635821844e-48, 1.9701509179759674e-48, 1.79128804738196e-48, 1.6145318239053106e-48, 1.4985845649553376e-48, 1.3791861942476244e-48, 1.3103072786569789e-48, 1.2960597140268965e-48, 1.2888920164521623e-48, 1.3090621670581823e-48, 1.3596579690050732e-48, 1.4293181576840035e-48, 1.4902181187198353e-48, 1.58128111190344e-48, 1.7142774341776054e-48, 1.849451078839559e-48, 1.9857164798316082e-48, 2.224741555234201e-48, 2.481242639023368e-48, 2.783441331068163e-48, 3.102554533375169e-48, 3.37863416149619e-48, 3.734490830043362e-48, 4.155609564181945e-48, 4.7129812850069816e-48, 5.373983856373061e-48, 5.934984372956836e-48, 6.722843267987003e-48, 7.646322481752521e-48, 8.807594799595915e-48, 9.954680872471326e-48, 1.192167501135572e-47, 1.4877801670641071e-47])


DARWIN_x = np.array([5.019370689312379, 5.306228169512471, 5.609043574278622, 5.928525562968069, 6.375933322070177, 6.696452348467524, 7.079338616709975, 7.62326696766099, 8.236043695280694, 8.843833564429385, 9.717000170093796, 10.561918511656884, 11.399977187384316, 12.379361383574121, 13.600891801378552, 15.16803188438747, 17.42013174867331, 19.934692350218008, 22.354086062117567, 25.181029986110094, 28.2774185189776, 32.40283377887805, 37.302713689880335, 42.8890560122588, 49.28006082439272, 55.4359409299913, 61.59641169357462, 68.98805892924462, 76.79568952028552, 87.03576629753617, 98.40897118641182, 110.63876625522457, 34.642022243875296, 30.287186490311168, 26.660852938128663, 23.817261749525567, 21.129729875249147, 18.62635779649876, 16.262076533338963, 39.91772306315006, 121.69152315570066, 133.49171740459508, 147.47562696141975, 167.30060349253176, 186.56303457287598, 207.16649886869052, 229.57495446998018, 257.6577564850282, 287.174755661089, 324.03673343277455, 358.984294041315, 398.7844039550788, 447.09058661575784, 498.5541814959728, 551.3227193036437, 628.0508407709013, 741.7900017396892, 850.0543883156714, 973.6153260731186])
DARWIN_y = np.array([2.130629806477718e-45, 1.224820655315024e-45, 7.030204300777559e-46, 4.028408537380879e-46, 1.964677006064814e-46, 1.1511387370380414e-46, 7.100458903028015e-47, 4.404312334030655e-47, 2.540401603542657e-47, 1.5313909115358246e-47, 8.55424610600025e-48, 5.378650395778611e-48, 3.5029311706455416e-48, 2.081958710931208e-48, 1.4330313855560013e-48, 9.77019632309727e-49, 6.57681969544458e-49, 4.895531964012349e-49, 3.88980311678816e-49, 3.182047424724544e-49, 2.8792450753002266e-49, 2.6279777112839797e-49, 2.519849553811706e-49, 2.6059871173848738e-49, 2.518438315293593e-49, 2.675138480281378e-49, 2.8272979030203218e-49, 2.914541755136244e-49, 3.101095313371427e-49, 3.3419110847260513e-49, 3.6717095324174995e-49, 3.9281247211468844e-49, 2.6538680504841104e-49, 2.8154438121710766e-49, 3.125074750509987e-49, 3.60041879283561e-49, 4.449571899509925e-49, 5.5992476424207415e-49, 7.913413262196974e-49, 2.564710676229181e-49, 4.2545603904500244e-49, 4.709250222009875e-49, 5.0254377126465064e-49, 5.524469941427123e-49, 6.054373607601679e-49, 6.6574449611159294e-49, 7.358620797853764e-49, 8.162174046329158e-49, 9.109436548305364e-49, 9.970632413045008e-49, 1.1110645990989102e-48, 1.2093097616340145e-48, 1.3721462449011014e-48, 1.540856878979035e-48, 1.664237311744973e-48, 1.8649397313198265e-48, 2.207477860885409e-48, 2.480632187232526e-48, 2.8707894272888654e-48])

LZ_x = np.array([9.062457499188575, 9.220762239351968, 9.408949511121085, 9.573306839991526, 9.740535194104634, 9.910684725078232, 10.083806460588198, 10.26418355985332, 10.495280803616081, 10.68301827306557, 10.869631278647097, 11.06406505400631, 11.31783715491855, 11.582204499073086, 11.879384527427359, 12.164668484072877, 12.448816511265258, 12.739601801237388, 13.037179390283704, 13.341707936109064, 13.653349802419344, 13.972271145487957, 14.298642002744403, 14.632636383432173, 14.984039807909479, 15.403043915583073, 15.783068478440425, 16.328811400290224, 17.100552752388232, 17.857154280506077, 18.701129022526526, 19.64160059987848, 20.80876690175167, 22.172913455039403, 23.626488460653643, 25.175354520428666, 26.825758566905275, 28.58435705864444, 30.45824282725351, 32.45497368440491, 34.582602904226604, 36.849711704005955, 39.26544385421016, 41.839542557412244, 44.582389744865004, 47.50504794921386, 50.61930492223199, 53.937721177530136, 57.473680649993064, 61.24144467626261, 65.25620951398356, 69.5341676318012, 74.09257301730568, 78.9498107663272, 84.12547123424935, 89.64042904841219, 95.51692730027786, 101.77866725692763, 108.4509039537147, 115.56054805362271, 123.13627438414925, 131.20863758947152, 139.81019536434337, 148.97563976675517, 158.74193713897225, 169.14847720128205, 180.23723191978368, 192.0529247889654, 204.64321121183093, 218.0588707050828, 232.35401170457737, 247.58628979707171, 263.8171402584484, 281.11202583629057, 299.54070077617996, 319.1774921565869, 340.1015996670507, 362.3974150387147, 386.15486241556215, 411.46976103913886, 438.4442117095634, 467.18700858150174, 497.81407795597806, 530.4489458377744, 565.2232361441737, 602.2772015744516, 641.76028928122, 683.8317436251209, 728.6612484439006, 776.4296114263037, 827.3294933510014, 881.5661851317642, 939.3584358028642, 1000.9393347841657, 1066.5572519842956, 1136.4768395335186, 1210.9800991865684, 1290.3675197004825, 1374.9592887747804, 1465.096584441981, 1561.1429511169774, 1663.4857658551432, 1772.5378007329757, 1888.738887652669, 2012.5576922852003, 2144.4936043066127, 2285.078751551266, 2434.8801462056176, 2594.501971698578, 2764.5880195120567, 2945.824285739847, 3138.941737867443, 3344.719262931765, 3563.986808951486, 3797.6287322979774, 4046.5873645076877, 4311.866812921621, 4594.537010480881, 4895.738031011937, 5216.684687406188, 5558.671431239386, 5923.077573592098, 6311.372848128112, 6725.1233388678065, 7165.99779656473, 7635.774369160596, 8136.347773464358, 8669.736936980207, 9238.093140705918, 9702.672793479753])
LZ_y = np.array([3.7660209321421486e-46, 3.3272469534959e-46, 2.955921732401844e-46, 2.640623068004886e-46, 2.35480246035342e-46, 2.1073340166090174e-46, 1.884211139938894e-46, 1.665732410225069e-46, 1.4616898458114063e-46, 1.2944828859370863e-46, 1.1553859666645903e-46, 1.0272189377929102e-46, 9.087394557707165e-47, 7.95694358628825e-47, 6.950765243182099e-47, 6.064245772385637e-47, 5.334489946845258e-47, 4.698757897470233e-47, 4.147002789075566e-47, 3.6818734701316773e-47, 3.2905894090312646e-47, 2.9370035452400935e-47, 2.6231448701718504e-47, 2.3490280534131103e-47, 2.0832642040958938e-47, 1.8382303558445193e-47, 1.6163101005888327e-47, 1.4413609604782962e-47, 1.2734299545997104e-47, 1.1324177482806384e-47, 1.0014263563301775e-47, 8.924421112821699e-48, 7.907601918738237e-48, 7.269652630656786e-48, 6.921834704931165e-48, 6.780238977128077e-48, 6.686383309980256e-48, 6.651123725425242e-48, 6.670333113366371e-48, 7.086796188253478e-48, 7.734679583239379e-48, 8.470244705855973e-48, 9.262395882996328e-48, 9.969238638082958e-48, 1.0818059955952311e-47, 1.1846853393655723e-47, 1.2898869579796034e-47, 1.404430618899095e-47, 1.5225451582415144e-47, 1.6450489688434675e-47, 1.776555292173679e-47, 1.9176523531957171e-47, 2.0699555842236988e-47, 2.2247101256760395e-47, 2.363610178180904e-47, 2.5260556064156264e-47, 2.682112155815501e-47, 2.838633733658023e-47, 3.015864182738379e-47, 3.2257944516571502e-47, 3.460302749827018e-47, 3.744113397835144e-47, 4.006661511706734e-47, 4.298306080134857e-47, 4.599321006543802e-47, 4.851375620627923e-47, 5.141899388976751e-47, 5.494535541324381e-47, 5.905319393244727e-47, 6.334623374577941e-47, 6.743078513212147e-47, 7.174421441318471e-47, 7.629688429646594e-47, 8.125553797597402e-47, 8.657806719206713e-47, 9.193937705534377e-47, 9.697789125555901e-47, 1.0288425288691294e-46, 1.1004588801613326e-46, 1.1787588828429587e-46, 1.2626300981618678e-46, 1.3485740568103144e-46, 1.4403679979983176e-46, 1.5398897401477287e-46, 1.6454967837568316e-46, 1.7558127648309218e-46, 1.853817084690303e-46, 1.9497797859656894e-46, 2.058610776163894e-46, 2.1955668086614577e-46, 2.3416343036339366e-46, 2.50463233314896e-46, 2.691884287334551e-46, 2.888966754532571e-46, 3.082646380626442e-46, 3.2845707765810824e-46, 3.496359186494509e-46, 3.734347295428118e-46, 3.988534638185465e-46, 4.2395966211017736e-46, 4.4978058942783326e-46, 4.7855226635092636e-46, 5.09654125597987e-46, 5.453925542150247e-46, 5.839176655989555e-46, 6.224654916618586e-46, 6.638771101629104e-46, 7.080437764053093e-46, 7.551487792421072e-46, 8.046137331410362e-46, 8.581433784047199e-46, 9.147944503565112e-46, 9.751853914654277e-46, 1.0405629323093646e-45, 1.1087235370676241e-45, 1.1824851138220237e-45, 1.2605478839264618e-45, 1.3444100879694582e-45, 1.433162474275645e-45, 1.5248393688713278e-45, 1.6169311164413894e-45, 1.720363369372376e-45, 1.8242636702009452e-45, 1.9409586236410805e-45, 2.0591711707927703e-45, 2.1845833594689536e-45, 2.3265628441295695e-45, 2.4647036702316236e-45, 2.6198467106919588e-45, 2.7499498603957025e-45])

interpolator_LZ=scp.interpolate.interp1d(LZ_x, LZ_y, kind='linear', fill_value="extrapolate")
interpolator_DARWIN=scp.interpolate.interp1d(DARWIN_x, DARWIN_y, kind='linear', fill_value="extrapolate")
interpolator_XENONnT=scp.interpolate.interp1d(XENONnT_x, XENONnT_y, kind='linear', fill_value="extrapolate")

panda4x=pd.read_csv('/home/js.sanchezl1/Proyecto_Teorico/PandaX-4T_2025.csv')
mass_panda=panda4x.iloc[:,0]
dd_xs_panda=panda4x.iloc[:,1]
interpolator_Panda4x=scp.interpolate.interp1d(mass_panda, dd_xs_panda, kind='linear', fill_value="extrapolate")

Fermi_data=pd.read_csv('/home/js.sanchezl1/Proyecto_Teorico/Fermi_data.csv')
mass_fermi=Fermi_data.iloc[:,0]
id_xs_fermi=Fermi_data.iloc[:,1]
interpolator_ID=scp.interpolate.interp1d(mass_fermi, id_xs_fermi, kind='linear', fill_value="extrapolate")

def constraints_direct_detection(blocks):
    mass_chi01=-1*blocks["MNIN"]["Mn(1,1)"]
    m_EtaI=blocks["MASS"]["etI"]
    dm_relic=blocks["MICROMEGAS"]["Omega h^2 (Dark matter relic density)"]
    neutron_SI=blocks["MICROMEGAS"]["neutron SI"]
    neutron_SI=neutron_SI*(dm_relic/0.12)
    #micromegas gives xs in pb^2 pass to cm^2
    neutron_SI=neutron_SI*1e-36  #pb^2 to cm^2

    if mass_chi01<m_EtaI:
        limit_fermion_dd=interpolator_Panda4x(mass_chi01)
        return (0<neutron_SI<limit_fermion_dd)
    else:
        limit_scalar_dd=interpolator_Panda4x(m_EtaI)
        return (0<neutron_SI<limit_scalar_dd)


def constraints_indirect_detection(blocks):
    mass_chi01=-1*blocks["MNIN"]["Mn(1,1)"]
    m_EtaI=blocks["MASS"]["etI"]
    dm_relic=blocks["MICROMEGAS"]["Omega h^2 (Dark matter relic density)"]
    id_xs=blocks["MICROMEGAS"]["sigma_v [cm^3/s] (indirect detection)"]
    id_xs=id_xs*((dm_relic/0.12)**2) #micromegas gives xs in pb pass to cm^3/s

    if mass_chi01<m_EtaI:
        limit_fermion_id=interpolator_ID(mass_chi01)
        return (0<id_xs<limit_fermion_id)
    else:
        limit_scalar_id=interpolator_ID(m_EtaI)
        return (0<id_xs<limit_scalar_id)


########################################################################################################################

############################ Lists for constraints ##########################################################
#lists may be defined as suited by the user

constraints_before_sp=[perturbativity, order_maj_masses, postitive_scalar_mass, valid_yukawa, small_lam5, stability_constraints, unitarity_constraints]

#constraints_after_sp_test=[any_DM_candidate, S_T_U_constraints,LFV_constraints,BR_higgs_inv,relic_density_constraint,constraints_direct_detection,constraints_indirect_detection]

constraints_after_sp_test=[any_DM_candidate,S_T_U_constraints, LFV_constraints, BR_higgs_inv, relic_density_constraint, constraints_direct_detection,constraints_indirect_detection]
########################################################################################################################

#dictionary for the parameter ranges

parameters_dictionary_Ma={"MINPAR": {
        "lambda1Input": lam_1fixed,  #Higgs self-coupling fixed
        "lambda2Input": [1e-6, 1],  #Eta doublet self-coupling
        "lambda3Input": [1e-6, 1],  #Higgs-EtaI trivial portal coupling
        "lambda4Input": [1e-6, 1],  #Higgs-EtaI non-trivial portal coupling
        "lambda5Input": [1e-12, 1e-2],  #Z2 term coupling
        "mEt2Input": [2.5e3, 1e8],  #EtaI mass term in potential
},
"MNIN":{
    "Mn(1,1)":[1e3, 1e8],   #Majorana mass of the first fermion
    "Mn(2,2)":[1e3, 1e8],  #Majorana mass of the second fermion
    "Mn(3,3)":[1e3, 1e8], #Majorana mass of the third fermion
},
"YNIN":{
    "ALL": casas_ibarra
},
"OTHER": {
        "v": 246.0,  #Higgs vacuum expectation value
        "mH": 125.0,  #Higgs mass,
        "Nu_1M":[1e-32, 1e-11],  #Neutrino mass scale for the first neutrino
}
}

#parameters for the likelihood

data_experimental={
    "MASS":{
        "hh":"0.0",
        "Chi_1":"0.0",  #Mass of the lightest fermion
        "Chi_2":"0.0",  #Mass of the second fermion
        "Chi_3":"0.0",  #Mass of the third fermion
        "etI":"0.0",  #Mass of the pseudo-scalar of eta
        "etR":"0.0",  #Mass of the scalar part of eta
        "etp":"0.0",  #Mass of the charged scalar of eta
        "Fv_1":"0.0",  #Lightest neutrino mass
        "Fv_2":"0.0",  #Second neutrino mass
        "Fv_3":"0.0",  #Third neutrino mass
    },
    "SPhenoLowEnergy":{
        "S-parameter (1-loop BSM)":"0.0",  #S-parameter
        "T-parameter (1-loop BSM)":"0.0",  #T-parameter
        "U-parameter (1-loop BSM)":"0.0",  #U-parameter
    },
    "FlavorKitLFV":{
        "BR(mu->e gamma)":"0.0",  #Muon to electron gamma decay
        "BR(tau->e gamma)":"0.0",  #Tau to electron gamma decay
        "BR(tau->mu gamma)":"0.0",  #Tau to muon gamma decay
        "BR(mu->3e)":"0.0",  #Muon to three electrons decay
        "BR(tau->3e)":"0.0",  #Tau to three electrons decay
        "BR(tau->3mu)":"0.0",  #Tau to three muons decay
        "BR(tau- -> e- mu+ mu-)":"0.0",  #Tau to electron muon muon decay
        "BR(tau- -> mu- e+ e-)":"0.0",  #Tau to muon electron electron decay
        "BR(tau- -> mu+ e- e-)":"0.0",  #Tau to muon electron electron decay
        #"BR(tau->e pi)":"0.0",  #Tau to electron pion decay
        #"BR(tau->e eta)":"0.0",  #Tau to electron eta decay
        #"BR(tau->e eta')":"0.0",  #Tau to electron eta' decay
        #"BR(tau->mu pi)":"0.0",  #Tau to muon pion decay
        #"BR(tau->mu eta)":"0.0",  #Tau to muon eta decay
        #"BR(tau->mu eta')":"0.0",  #Tau to muon eta' decay
        "CR(mu-e, Ti)":"0.0",  #Muon-electron conversion in Titanium
        "CR(mu-e, Pb)":"0.0",  #Muon-electron conversion in Lead
        "CR(mu-e, Au)":"0.0",  #Muon-electron conversion in Gold
        "BR(Z->e mu)":"0.0",  #Z boson to electron muon decay
        "BR(Z->e tau)":"0.0",  #Z boson to electron tau decay
        "BR(Z->mu tau)":"0.0"   #Z boson to muon tau decay
   },
    "DECAY 25": {
        "WIDTH":[4.6e-3,2.6e-3],  #Higgs width
        "BR(hh -> Chi_1 Chi_1)":"0.0",  #Higgs decay to two fermions
        "BR(hh -> Chi_2 Chi_2)":"0.0",  #Higgs decay to two fermions
        "BR(hh -> Chi_3 Chi_3)":"0.0",  #Higgs decay to two fermions
        "BR(hh -> etI etI)":"0.0",  #Higgs decay to two pseudo-scalars
        "BR(hh -> etR etR)":"0.0"  #Higgs decay to two scalars
    },
    "MICROMEGAS":{
        "Xf (Freeze-out temperature)":"0.0",  #Freeze-out temperature
        "Omega h^2 (Dark matter relic density)":"0.0",  #Dark matter relic density
        "sigma_v [cm^3/s] (indirect detection)":"0.0",  #Dark matter annihilation cross-section
        "neutron SI":"0.0",  #Neutron spin-independent cross-section
    }
}

##########################################################################################################################


#Paths

#path to the Spheno executable

spheno_exec_path='/home/js.sanchezl1/SPheno/bin/SPhenoScotogenic_v1'

#path to the SPheno input file

spheno_input_path="/home/js.sanchezl1/SPheno/Scotogenic_v1/Input_Files/LesHouches.in.Scotogenic_v1"

micromegas_file_path='/home/js.sanchezl1/micromegas_6.0.5/Scotogenic_v1/CalcOmega_with_DDetection_MOv5_4'


###########################################################################################################################

#command with the sampler file

sampling= smmcmc(
    SphenoBlocksDict=parameters_dictionary_Ma,
    ExpectedDataDict=data_experimental,
    ConstraintsBeforeSPheno=constraints_before_sp,
    ConstraintsAfterSPheno=constraints_after_sp_test,
    SPhenoFilePath=spheno_exec_path,
    SPhenoInputFilePath=spheno_input_path,
    UseMicrOmegas=True,
    MicrOmegasFilePath=micromegas_file_path,
    nWalkers=200,  #Number of walkers
    LikelihoodThreshold=0.9,  #Likelihood threshold for the acceptance of the points
    AcceptedPoints=20000,
    OutputMCMCfile="/home/js.sanchezl1/Proyecto_Teorico/template7.csv",
    Steps= None,
    Stretch= 1.8, 
    LogParameterization= True,
    StrictParametersRanges=True,  #If True, the parameters will be sampled within the ranges defined in the parameters dictionary
    WriteOnlyAccepted=True,  #If True, only the accepted points will be written to the output file
)

#Run the sampler
sampling.run()